from pathlib import Path
import argparse

def count_defects(frame, data):
    vacancy = data.attributes['WignerSeitz.vacancy_count']
    interstitial = data.attributes['WignerSeitz.interstitial_count']
    
    if vacancy != interstitial:
        print(f"Warning: Timestep {frame}: Vacancies({vacancy}) != Interstitials({interstitial})", flush=True)
    
    data.attributes.update(
        Frenkel_pairs=interstitial,
        Timestep=frame
    )

def analyze_cascade(work_dir, cascade_dir, energy, output_dir=None):
    try:
        from ovito.io import import_file
        from ovito.modifiers import WignerSeitzAnalysisModifier
    except ImportError as error:
        raise RuntimeError("Install OVITO in the analysis Python environment or configure ovito_command.") from error

    xyz_path = Path(cascade_dir) / "dump.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"Input file not found: {xyz_path}")
    pipeline = import_file(str(xyz_path))
    if pipeline.source.num_frames < 2:
        raise ValueError(f"Expected a reference and final trajectory frame: {xyz_path}")
    pipeline.modifiers.append(WignerSeitzAnalysisModifier(reference_frame=0))
    pipeline.modifiers.append(count_defects)
    final_frame = pipeline.source.num_frames - 1
    data = pipeline.compute(final_frame)
    fp = data.attributes['Frenkel_pairs']
    summary = f"{final_frame}\t{fp}\n"
    contents = "Timestep\tFrenkel_pairs\n" + summary
    if output_dir is not None:
        (Path(output_dir) / "final_defect_summary.txt").write_text(contents, encoding="utf-8")
    (Path(work_dir) / "final_defect_summary.txt").write_text(contents, encoding="utf-8")

    history_path = Path(work_dir) / "defect_evolution_history.txt"
    header = not history_path.exists() or history_path.stat().st_size == 0
    
    with history_path.open("a", encoding="utf-8") as f:
        if header:
            f.write("Energy(eV)\tTimestep\tFrenkel_pairs\n")
        f.write(f"{energy:.4f}\t{summary}" if energy is not None else f"NaN\t{summary}")
    
    print(f"Results saved to:\n- {Path(work_dir)/'final_defect_summary.txt'}\n- {history_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze atomic cascade defects")
    parser.add_argument("--work_dir", required=True, help="Base directory of simulation data")
    parser.add_argument("--cascade_dir", required=True, help="Cascade directory of simulation data")
    parser.add_argument("--energy", type=float, required=True, help="PKA energy in eV")
    parser.add_argument("--output_dir", help="Also write a case-specific defect summary")
    args = parser.parse_args()
    
    analyze_cascade(args.work_dir, args.cascade_dir, args.energy, args.output_dir)
