import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import WignerSeitzAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier, AssignColorModifier

def wigner_seitz_analysis(input_file, output):
    pipeline = import_file(input_file)
    
    wigner_seitz = WignerSeitzAnalysisModifier(output_displaced=False)
    pipeline.modifiers.append(wigner_seitz)
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="Occupancy == 1"))
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    last_frame_index = pipeline.source.num_frames - 1
    final_data = pipeline.compute(last_frame_index)
    output = 1 if final_data.particles.count > 0 else 0
    
    pipeline.remove_from_scene()
    del pipeline
    
    return output

def gif_generator(input_file, output_file):
    pipeline = import_file(input_file, multiple_frames=True)
    wigner_seitz = WignerSeitzAnalysisModifier(
        output_displaced=False
    )
    pipeline.modifiers.append(wigner_seitz)
    
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="Occupancy == 1"))
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="Occupancy > 1"))
    pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.0, 0.0)))
    
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="Occupancy == 0"))
    pipeline.modifiers.append(AssignColorModifier(color=(0.0, 0.0, 1.0)))
    
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Front)
    vp.zoom_all()

    renderer = TachyonRenderer()

    try:
        vp.render_anim(
            filename=output_file,
            size=(800, 800), 
            renderer=renderer,
            fps=5, 
            range=(0, pipeline.source.num_frames - 1) 
        )
        print(f"GIF 动画已保存到：{os.path.abspath(output_file)}")
    except Exception as e:
        print(f"保存 GIF 时发生错误：{e}")
    finally:
        pipeline.remove_from_scene() 
        del pipeline 

def main():
    parser = argparse.ArgumentParser(description="根据输入的轨迹文件生成 GIF 动图")
    parser.add_argument("--input_file", type=str, default="dump.xyz", help="轨迹文件名")
    parser.add_argument("--output_file", type=str, default="dump.gif", help="生成的 GIF 文件名")

    args = parser.parse_args()
    gif_generator(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
    
