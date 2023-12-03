import gradio as gr
import numpy as np
from PIL import Image
from motion_brush_utils import MotionBrush

motion_brush = MotionBrush()

def reset_max_steps_to_replace_if_invalid(max_steps_to_replace, num_inference_steps):
    if max_steps_to_replace > num_inference_steps:
        return num_inference_steps
    else:
        return max_steps_to_replace

def animate_image(
    image,
    num_inference_steps,
    max_steps_to_replace,
    num_frames,
    fps,
    motion_bucket_id,
    seed,
):

    source_image = image['background'][..., :3]
    mask = image['layers'][0][..., -1:]
    mask = (mask > 128).astype(np.float32)

    if mask.sum() == 0:
        mask = None

    gif_path = motion_brush(
        source_image, 
        mask,
        max_steps_to_replace=int(max_steps_to_replace),
        num_frames=int(num_frames),
        num_inference_steps=int(num_inference_steps),
        fps=int(fps),
        motion_bucket_id=int(motion_bucket_id),
        seed=int(seed),
    )
    return gif_path

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Input")
            input_image = gr.ImageEditor(height=512)
        with gr.Column():
            gr.Markdown("## Output")
            output_image = gr.Image(height=512)

    with gr.Row():
        gr.Markdown("## Parameters")

    with gr.Row():
        with gr.Column():
            num_inference_steps = gr.Slider(minimum=20, maximum=50, step=5, value=25, label="Number of Inference Steps")
            max_steps_to_replace = gr.Slider(minimum=1, maximum=50, step=1, value=25, label="Max Steps to Replace")
            motion_bucket_id = gr.Slider(minimum=0, maximum=300, step=1, value=30, label="Motion Bucket ID")
        num_frames = gr.Textbox(label="Number of Frames", value=25)
        fps = gr.Textbox(label="FPS", value=8)
        seed = gr.Textbox(label="Seed", value=42)

    submit_btn = gr.Button(value="Generate")
    
    # Now gr.Examples has a bug, when you submit the task, the image will be reset and the mask will disappear
    # dummy_image_to_show = gr.Image(height=512, visible=False)
    # gr.Examples(
    #     examples=[
    #         [{
    #             'background': 'figures/test_image_cloud_resized.jpg',
    #             'layers': [np.zeros_like(np.array(Image.open('figures/test_image_cloud_resized.jpg').convert('RGBA')))],
    #             'composite': np.zeros_like(np.array(Image.open('figures/test_image_cloud_resized.jpg').convert('RGBA'))),
    #         }, 'figures/test_image_cloud_resized.jpg']
    #     ],
    #     inputs=[input_image, dummy_image_to_show],
    # )

    submit_btn.click(
        animate_image, 
        inputs=[
            input_image,
            num_inference_steps,
            max_steps_to_replace,
            num_frames,
            fps,
            motion_bucket_id,
            seed,
        ], 
        outputs=[output_image]
    )

    num_inference_steps.change(
        reset_max_steps_to_replace_if_invalid,
        inputs=[
            max_steps_to_replace,
            num_inference_steps
        ],
        outputs=[
            max_steps_to_replace
        ]
    )

    max_steps_to_replace.change(
        reset_max_steps_to_replace_if_invalid,
        inputs=[
            max_steps_to_replace,
            num_inference_steps
        ],
        outputs=[
            max_steps_to_replace
        ]
    )

demo.launch(share=True)