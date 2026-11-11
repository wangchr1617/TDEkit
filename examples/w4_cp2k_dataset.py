"""Audit the W4 CP2K relabel batch and stage a standalone NEP candidate dataset."""

import argparse
import csv
import hashlib
import json
from itertools import islice, zip_longest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread, read

from tdekit.cp2k_labels import read_cp2k_label
from tdekit.nep_dataset import (
    DatasetFrame, PartitionEntry, audit_partitions, geometry_fingerprint, write_nep_dataset,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hash(path, expected):
    require(digest(path) == expected, f"SHA-256 mismatch: {path}")


def json_data(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_atoms(metadata):
    return Atoms(metadata["symbols"], positions=np.reshape(metadata["coordinates"], (-1, 3)),
                 cell=np.reshape(metadata["cell"], (3, 3)), pbc=True)


def check_same_geometry(first, second, context):
    require(first.get_chemical_symbols() == second.get_chemical_symbols()
            and np.allclose(first.cell, second.cell, rtol=0, atol=1e-10)
            and np.allclose(first.positions, second.positions, rtol=0, atol=1e-10),
            f"Frozen source geometry mismatch: {context}")


def compare_label(label, receipt, case):
    observed = (label.energy_ev, label.forces_ev_angstrom.ravel(),
                label.stress_gpa.ravel(), label.virial_ev.ravel())
    fields = ("energy_eV", "force_eV_A", "stress_GPa", "virial_eV")
    maximum = {}
    for key, value in zip(fields, observed):
        expected = np.asarray(receipt[key])
        difference = np.abs(np.asarray(value) - expected)
        require(np.shape(value) == expected.shape and np.isfinite(difference).all()
                and np.max(difference) <= 1e-10, f"Label differs from W4 receipt: {case} {key}")
        maximum[key] = float(np.max(difference))
    return maximum


def read_case(folder, frozen, hashes):
    for filename in ("geometry.json", "input.inp"):
        check_hash(folder / filename, frozen[filename])
    metadata = json_data(folder / "geometry.json")
    require(metadata["name"] == folder.name, f"Invalid case ID: {folder}")
    atoms = source_atoms(metadata)
    require(len(atoms) == metadata["natoms"], f"Invalid atom count: {folder}")
    raw = (folder / "output.log").read_text(encoding="utf-8", errors="replace")
    require(any(line.split()[-2:] == ["version", "2025.1"] for line in raw.splitlines()
                   if "CP2K| version string:" in line), f"Unexpected CP2K version: {folder}")
    label = read_cp2k_label(raw, atoms, exit_code=int((folder / "exit_code").read_text()))
    acceptance = json_data(folder / "acceptance.json")
    require(acceptance["PASS"] is True and acceptance["result"]["name"] == folder.name
            and acceptance["result"]["natoms"] == len(atoms), f"Invalid receipt: {folder}")
    differences = compare_label(label, acceptance["result"], folder.name)
    for filename, expected in hashes.items():
        check_hash(folder / filename, expected)
    return metadata, atoms, label, differences


def ordered_w4_key(atoms):
    # Matches the original ASI candidate score key. Export uses the library's
    # separate order-insensitive geometry fingerprint for split auditing.
    return hashlib.sha256(
        np.round(atoms.cell.array, 8).astype("<f8").tobytes()
        + "".join(atoms.get_chemical_symbols()).encode()
        + np.round(atoms.get_scaled_positions(wrap=True), 6).astype("<f8").tobytes()
    ).hexdigest()


def family(atoms):
    value = atoms.info.get("Config_type", atoms.info.get("config_type"))
    require(isinstance(value, str) and bool(value.strip()), "Missing Iter7 configuration family")
    return "iter7:" + value


def load_candidates(root, workdir, evidence, frozen, train_path, train_atoms):
    scored_dir = root.parent / "post_training_resume_03/accepted_385521/results"
    scored = scored_dir / "candidate_scores.csv"
    check_hash(scored, json_data(scored_dir / "output_hashes.json")["results/candidate_scores.csv"])
    unique = {}
    scores = csv_rows(scored)
    require(len(scores) == 446, "Expected the 446 frozen ASI score rows")
    for row in sorted(scores, key=lambda row: float(row["gamma_max"]), reverse=True):
        if row["exceeds_coverage"] == "True":
            unique.setdefault(row["geometry_sha256"], row)
    eligible = {row["candidate_id"]: row for row in unique.values()}

    snapshot = evidence / "production_snapshot"
    rows = csv_rows(snapshot / "local_audit.csv")
    check_hash(root / "batches.json", frozen["batches.json"])
    batches = json_data(root / "batches.json")
    planned = {case["name"]: (batch["batch"], case["natoms"])
               for batch in batches for case in batch["candidates"]}
    require(len(rows) == len(eligible) == len(planned) == 168
            and {row["id"] for row in rows} == set(planned) == set(eligible)
            and {path.name for path in (snapshot / "candidates").iterdir()} == set(planned),
            "Candidate inventory disagrees with the frozen plan")
    augmentation = workdir / "tde/te_augmentation_20261104"
    check_hash(augmentation / "split_manifest.json",
               json_data(augmentation / "checksums.json")["split_manifest.json"])
    split = json_data(augmentation / "split_manifest.json")
    source_roles = {row["case"]: row for row in split["new_cases"]}
    frames = []
    discrepancies = []
    source_hashes = {train_path: digest(train_path)}
    for row in rows:
        name = row["id"]
        require((int(row["batch"]), int(row["atoms"])) == planned[name], f"Batch mismatch: {name}")
        rel = f"candidates/{name}/"
        expected = {file: frozen[rel + file] for file in ("geometry.json", "input.inp")}
        files = {file: row[key + "_sha256"] for key, file in (
            ("geometry", "geometry.json"), ("input", "input.inp"),
            ("output", "output.log"), ("acceptance", "acceptance.json"))}
        meta, atoms, label, differences = read_case(snapshot / rel, expected, files)
        score = eligible[name]
        source = meta["source"]
        require(source == score and score["original_role"] == "train", f"ASI source mismatch: {name}")
        path = (workdir / source["source_path"]).resolve()
        require(path.is_relative_to(workdir), f"Source escaped workdir: {name}")
        if path not in source_hashes:
            source_hashes[path] = digest(path)
        require(source_hashes[path] == score["source_sha256"] == row["source_sha256"],
                f"Source hash mismatch: {name}")
        frame_index = int(source["source_frame"])
        if path == train_path:
            require(0 <= frame_index < len(train_atoms), f"Invalid source frame: {name}")
            original = train_atoms[frame_index]
            group = family(original)
            require(group == "iter7:" + score["family"], f"Source family mismatch: {name}")
        else:
            role = source_roles[name]
            require(role["partition"] == "train" and frame_index == 0
                    and path == workdir / "tde/te_augmentation_20261104/cases" / name / "reference_data/POSCAR",
                    f"Unexpected source: {name}")
            original = read(path, format="vasp")
            group = f'w4-recovery:seed-{role["source_seed"]}'
        check_same_geometry(atoms, original, name)
        require(ordered_w4_key(original) == row["geometry_fingerprint"] == score["geometry_sha256"],
                f"Original W4 geometry fingerprint mismatch: {name}")
        require(len(atoms) == int(row["atoms"]) and abs(label.energy_ev - float(row["energy_eV"])) <= 1e-10,
                f"Audit index disagrees with recomputed label: {name}")
        frames.append(DatasetFrame(name, "candidate", group, atoms, label, {
            "source_path": source["source_path"], "source_frame": frame_index,
            "source_sha256": source_hashes[path], "w4_geometry_sha256": score["geometry_sha256"],
            "result_path": rel.rstrip("/"), "files_sha256": files,
            "cp2k_version": "2025.1", "original_role": "train",
        }))
        discrepancies.append(differences)
    return frames, discrepancies, split


def raw_row(path, index):
    with path.open(encoding="utf-8") as stream:
        return np.array([float(x) for x in next(islice(stream, index, None)).split()])


def bridge_checks(root, evidence, frozen, public_db, public_rows):
    members = {row["name"]: row["sha256"] for row in
               json_data(root.parent / "results/zip_validation.json")["members"] if not row["directory"]}
    partitions = {(row["source_group"], int(row["frame"])): row for row in public_rows}
    results = []
    for index in range(1, 7):
        name = f"bridge_{index:02d}"
        rel = f"bridge/{name}/"
        expected = {file: frozen[rel + file] for file in ("geometry.json", "input.inp")}
        meta, atoms, label, differences = read_case(evidence / rel, expected, {})
        check_hash(evidence / f"state/{name}.json", frozen[f"state/{name}.json"])
        compare_label(label, json_data(evidence / f"state/{name}.json")["result"], name)
        group, frame = meta["source"]["group"], meta["source"]["frame"]
        raw = {}
        for key in ("box", "coord", "energy", "force", "virial", "type"):
            path = public_db / group / f"{key}.raw"
            check_hash(path, members[f"Database/{group}/{key}.raw"])
            if key != "type":
                raw[key] = raw_row(path, frame)
        symbols = [("Ge", "Te")[int(value)] for value in
                   (public_db / group / "type.raw").read_text().split()]
        original = Atoms(symbols, positions=raw["coord"].reshape(-1, 3),
                         cell=raw["box"].reshape(3, 3), pbc=True)
        check_same_geometry(atoms, original, name)
        require(public_geometry_key(original) == partitions[group, frame]["geometry_sha256"],
                f"Public bridge geometry fingerprint mismatch: {name}")
        for key in ("energy", "force", "virial"):
            require(np.array_equal(raw[key], meta["reference"][key]), f"Public label mismatch: {name} {key}")
        df = label.forces_ev_angstrom.ravel() - raw["force"]
        dv = label.virial_ev.ravel() - raw["virial"]
        opposite = -label.virial_ev.ravel() - raw["virial"]
        rmse = lambda x: float(np.sqrt(np.mean(x**2)))
        metrics = {
            "case_id": name, "source_group": group, "source_frame": frame,
            "partition": partitions[group, frame]["partition"], "natoms": len(atoms),
            "composition": atoms.get_chemical_formula(),
            "force_rmse_ev_angstrom": rmse(df), "max_force_component_ev_angstrom": float(np.max(abs(df))),
            "virial_rmse_ev_atom": rmse(dv) / len(atoms),
            "opposite_virial_rmse_ev_atom": rmse(opposite) / len(atoms),
            "energy_residual_ev_atom": float((label.energy_ev - raw["energy"][0]) / len(atoms)),
            "receipt_max_abs_difference": differences,
        }
        require(metrics["force_rmse_ev_angstrom"] <= 0.05
                and metrics["max_force_component_ev_angstrom"] <= 0.2
                and metrics["virial_rmse_ev_atom"] <= 0.03
                and abs(metrics["energy_residual_ev_atom"]) <= 0.03,
                f"Bridge numerical gate failed: {name}")
        results.append(metrics)
    pair_residual = abs(results[2]["energy_residual_ev_atom"] - results[3]["energy_residual_ev_atom"])
    require(results[2]["composition"] == results[3]["composition"] and pair_residual <= 0.005
            and sum(r["virial_rmse_ev_atom"]**2 for r in results)
            < sum(r["opposite_virial_rmse_ev_atom"]**2 for r in results),
            "Bridge joint energy or virial-sign gate failed")
    return results, pair_residual


def public_geometry_key(atoms):
    # Reuse the public archive's own fingerprint convention, including its
    # Ge/Te type IDs, for comparisons against the historical frame manifest.
    cell = atoms.cell.array
    frac = np.round(np.mod(atoms.positions @ np.linalg.inv(cell), 1), 6)
    frac[frac == 1] = 0
    types = np.array([{"Ge": 0, "Te": 1}[e] for e in atoms.get_chemical_symbols()])
    order = np.lexsort((frac[:, 2], frac[:, 1], frac[:, 0], types))
    return hashlib.sha256(np.round(cell, 6).astype("<f8").tobytes()
                          + types[order].astype("<i4").tobytes()
                          + frac[order].astype("<f8").tobytes()).hexdigest()


def w4_audits(workdir, frames, public, split, train_atoms, fixed_test):
    # Candidate frames are all training-role; audit proposed use without promoting them.
    candidates = [PartitionEntry(f.case_id, "train", f.source_group,
                                 f.partition_entry().geometry_sha256) for f in frames]
    train = [DatasetFrame(f"iter7-train-{i}", "train", family(a), a) for i, a in enumerate(train_atoms)]
    tests = [DatasetFrame(f"iter7-test-{i}", "test", family(a), a) for i, a in enumerate(fixed_test)]
    sealed = []
    augmentation = workdir / "tde/te_augmentation_20261104"
    augmentation_hashes = json_data(augmentation / "checksums.json")
    for row in split["new_cases"]:
        if row["partition"] == "test":
            case = row["case"]
            relative = f"cases/{case}/reference_data/POSCAR"
            path = augmentation / relative
            check_hash(path, augmentation_hashes[relative])
            sealed.append(DatasetFrame(case, "test", f'w4-recovery:seed-{row["source_seed"]}',
                                       read(path, format="vasp")))
    require(len(sealed) == 2, "Expected two frozen sealed recovery frames")
    public_entries = [
        PartitionEntry(f'{r["source_group"]}:{r["frame"]}',
                       "test" if r["partition"] == "heldout" else r["partition"],
                       r["source_group"], r["geometry_sha256"])
        for r in public if r["partition"] in {"train", "development", "heldout"}
    ]
    comparison = audit_partitions([*candidates, *tests, *sealed])
    shared_groups = set(comparison["shared_source_groups"])
    public_test_keys = {row.geometry_sha256 for row in public_entries if row.partition == "test"}
    return {
        "candidate_vs_w4_test": comparison,
        "candidate_ids_in_shared_w4_groups": [r.case_id for r in candidates if r.source_group in shared_groups],
        "legacy_iter7_train_vs_fixed_test": audit_partitions([*train, *tests]),
        "public_historical_split": audit_partitions(public_entries),
        "candidate_ids_matching_public_heldout_geometry": [
            frame.case_id for frame in frames if public_geometry_key(frame.atoms) in public_test_keys
        ],
        "group_policy": "Iter7 Config_type is a family proxy, not a verified parent trajectory ID. "
                        "Recovery structures use source seed. Public structures use archived source group.",
    }


def verify_export(output, frames):
    xyz = output / "candidate.xyz"
    manifest = json_data(output / "manifest.json")
    check_hash(xyz, manifest["files"][xyz.name])
    force_error = position_error = 0.0
    exported = iread(xyz, index=":", format="extxyz")
    for frame, row, atoms in zip_longest(frames, manifest["frames"], exported):
        require(frame is not None and row is not None and atoms is not None,
                "Exported frame count disagrees with manifest")
        require(row["case_id"] == frame.case_id
                and row["geometry_sha256"] == geometry_fingerprint(atoms),
                f"Exported geometry fingerprint differs: {frame.case_id}")
        require(atoms.get_chemical_symbols() == frame.atoms.get_chemical_symbols()
                and np.array_equal(atoms.cell.array, frame.atoms.cell.array)
                and np.array_equal(atoms.pbc, frame.atoms.pbc),
                f"Exported species, cell or periodicity changed: {frame.case_id}")
        label = frame.label
        require(atoms.get_potential_energy() == label.energy_ev
                and np.array_equal(atoms.info["virial"], label.virial_ev),
                f"Exported energy or virial changed: {frame.case_id}")
        force_error = max(force_error, float(np.max(abs(atoms.get_forces() - label.forces_ev_angstrom))))
        position_error = max(position_error, float(np.max(abs(atoms.positions - frame.atoms.positions))))
    require(force_error <= 5.01e-9 and position_error <= 5.01e-9,
            "Exported positions or forces exceed extxyz printing precision")
    return {"force_max_abs_difference_ev_angstrom": force_error,
            "position_max_abs_difference_angstrom": position_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="W4 cp2k_relabel_168_20261110 directory")
    parser.add_argument("public_db", type=Path, help="Extracted public Database directory")
    parser.add_argument("output", type=Path, help="Fresh output directory")
    args = parser.parse_args()
    root = args.batch.resolve()
    workdir = root.parents[2]
    evidence = root / "production_recovery/remote_evidence"
    checksums = (evidence / "production_recovery/SHA256SUMS").read_text().splitlines()
    frozen = {name: checksum for checksum, name in (line.split(maxsplit=1) for line in checksums)}
    iter7 = workdir / "nep_train/iter7/_shared"
    expected = json_data(workdir / "potentials/iter7_freeze.json")["input_hashes"]
    for filename in ("train.xyz", "test.xyz"):
        check_hash(iter7 / filename, expected[f"workdir/nep_train/iter7/_shared/{filename}"])
    train_atoms = read(iter7 / "train.xyz", index=":", format="extxyz")
    fixed_test = read(iter7 / "test.xyz", index=":", format="extxyz")
    require((len(train_atoms), len(fixed_test)) == (443, 399), "Unexpected Iter7 data inventory")
    frames, differences, split = load_candidates(root, workdir, evidence, frozen, iter7 / "train.xyz", train_atoms)
    public_file = root.parent / "data/frame_manifest.csv"
    check_hash(public_file, json_data(public_file.with_name("input_hashes.json"))[public_file.name])
    public = csv_rows(public_file)
    bridges, pair_residual = bridge_checks(root, evidence, frozen, args.public_db.resolve(), public)
    splits = w4_audits(workdir, frames, public, split, train_atoms, fixed_test)
    splits["bridge_groups_shared_with_public_heldout"] = sorted(
        {row["source_group"] for row in bridges}
        & {row["source_group"] for row in public if row["partition"] == "heldout"}
    )
    report = {
        "candidates": len(frames), "candidate_atoms": sum(len(frame.atoms) for frame in frames),
        "candidate_receipt_max_abs_difference": {key: max(r[key] for r in differences) for key in differences[0]},
        "bridges": bridges, "bridge_pair_energy_residual_ev_atom": pair_residual,
        "source_group_audit": splits, "export_role": "candidate",
    }
    write_nep_dataset(args.output, frames)
    report["candidate_xyz_round_trip"] = verify_export(args.output, frames)
    (args.output / "w4_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidates": len(frames), "bridges": len(bridges), "output": str(args.output)}))


if __name__ == "__main__":
    main()
