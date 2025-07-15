from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
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

def analyze_cascade(base_dir, cascade_dir, energy):
    base_path = Path(base_dir)
    xyz_path = base_path / cascade_dir / "dump.xyz"

    if not xyz_path.exists():
        raise FileNotFoundError(f"Input file not found: {xyz_path}")

    pipeline = import_file(str(xyz_path))
    pipeline.source.data.cell = pipeline.compute(0).cell

    pipeline.modifiers.append(WignerSeitzAnalysisModifier(reference_frame=0))
    pipeline.modifiers.append(count_defects)

    final_frame = pipeline.source.num_frames - 1
    data = pipeline.compute(final_frame)
    fp = data.attributes['Frenkel_pairs']

    summary = f"{final_frame}\t{fp}\n"
    (base_path / "final_defect_summary.txt").write_text(
        "Timestep\tFrenkel_pairs\n" + summary,
        encoding="utf-8"
    )

    history_path = base_path / "defect_evolution_history.txt"
    header = not history_path.exists() or history_path.stat().st_size == 0
    
    with history_path.open("a", encoding="utf-8") as f:
        if header:
            f.write("Energy(eV)\tTimestep\tFrenkel_pairs\n")
        f.write(f"{energy:.4f}\t{summary}" if energy else f"NaN\t{summary}")
    
    print(f"Results saved to:\n- {base_path/'final_defect_summary.txt'}\n- {history_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze atomic cascade defects")
    parser.add_argument("--base_dir", required=True, help="Base directory of simulation data")
    parser.add_argument("--cascade_dir", required=True, help="Cascade directory of simulation data")
    parser.add_argument("--energy", type=float, required=True, help="PKA energy in eV")
    args = parser.parse_args()
    
    analyze_cascade(args.base_dir, args.cascade_dir, args.energy)