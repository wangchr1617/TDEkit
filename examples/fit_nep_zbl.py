"""Fit one legacy NEP-ZBL pair and report training and holdout errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tdekit.zbl import fit_amplitudes, read_flexible_zbl, write_candidate

COLUMNS = ("distance_angstrom", "target_energy_ev", "target_force_ev_per_angstrom")


def read_reference(path):
    data = {"train": [], "holdout": []}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                data[row["split"]].append([float(row[column]) for column in COLUMNS])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid split or numeric columns in {path}: {row}") from error
    if not all(data.values()):
        raise ValueError("Both train and holdout observations are required")
    if not all(np.isfinite(rows).all() for rows in data.values()):
        raise ValueError("Reference values must be finite")
    train, holdout = (np.asarray(data[split]) for split in ("train", "holdout"))
    if set(train[:, 0]) & set(holdout[:, 0]):
        raise ValueError("Training and holdout distances must be disjoint")
    return train, holdout


def metrics(model, pair, data):
    distance, target_energy, target_force = data.T
    energy, force = model.evaluate(pair, distance)
    return {
        "energy_rmse_ev": float(np.sqrt(np.mean((energy - target_energy)**2))),
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean((force - target_force)**2))),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", nargs=2, required=True)
    parser.add_argument("--energy-scale", type=float, required=True)
    parser.add_argument("--force-scale", type=float, required=True)
    args = parser.parse_args()

    parameters = read_flexible_zbl(args.model)
    pair = next((pair for pair in parameters if sorted(pair) == sorted(args.pair)), None)
    if pair is None:
        parser.error(f"Pair {args.pair} is absent from the model")
    train, holdout = read_reference(args.reference)
    baseline = parameters[pair]
    candidate = fit_amplitudes(
        baseline, pair, *train.T, energy_scale=args.energy_scale,
        force_scale=args.force_scale,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    parameters[pair] = candidate
    model_path = args.output / "nep_candidate.txt"
    write_candidate(args.model, model_path, parameters)
    report = {
        "pair": pair,
        "source_sha256": digest(args.model),
        "reference_sha256": digest(args.reference),
        "candidate_sha256": digest(model_path),
        "energy_scale_ev": args.energy_scale,
        "force_scale_ev_per_angstrom": args.force_scale,
        "baseline_parameters": baseline.parameters(),
        "candidate_parameters": candidate.parameters(),
    }
    for split, data in (("train", train), ("holdout", holdout)):
        report[split] = {
            "n": len(data), "baseline": metrics(baseline, pair, data),
            "candidate": metrics(candidate, pair, data),
        }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
