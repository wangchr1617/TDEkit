from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from pathlib import Path
import os

def count_defects(frame, data):
    """Count and record vacancy and interstitial defects"""
    vacancy = data.attributes['WignerSeitz.vacancy_count']
    interstitial = data.attributes['WignerSeitz.interstitial_count']
    
    if vacancy != interstitial:
        print(f"Warning: At timestep {frame}, vacancy count ({vacancy}) != interstitial count ({interstitial})")
    
    data.attributes.update(Frenkel_pairs=interstitial, Timestep=frame)
    
base_dir = Path(__file__).parent
input_xyz = base_dir / "cascade" / "dump.xyz"

if not os.path.exists(input_xyz):
    print(f"Error: File not found - {input_xyz}")
    exit(1)

pipeline = import_file(str(input_xyz))
data0 = pipeline.compute(0)
pipeline.source.data.cell = data0.cell

pipeline.modifiers.append(WignerSeitzAnalysisModifier(reference_frame=0))
pipeline.modifiers.append(count_defects)

last_frame = pipeline.source.num_frames - 1
data = pipeline.compute(last_frame)
fp = data.attributes['Frenkel_pairs']

summary_path = base_dir / "final_defect_summary.txt"
summary_path.write_text(f"Timestep\tFrenkel_pairs\n{last_frame}\t{fp}\n", encoding="utf-8")

energy = None
cascade_script = base_dir / "cascade.py"
if cascade_script.exists():
    for line in cascade_script.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("energy"):
            try:
                energy = float(line.split("=")[1].split("#")[0].strip())
            except ValueError:
                pass

history_path = base_dir / "defect_evolution_history.txt"
header = not os.path.exists(history_path) or os.path.getsize(history_path) == 0

with open(history_path, "a", encoding="utf-8") as f:
    if header:
        f.write("Energy(eV)\tTimestep\tFrenkel_pairs\n")
    if energy is not None:
        f.write(f"{energy:.4f}\t{last_frame}\t{fp}\n")
    else:
        f.write(f"NaN\t{last_frame}\t{fp}\n")

print(f"Processing completed. Results saved to: {summary_path} and {history_path}")