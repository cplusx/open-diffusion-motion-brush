import gradio as gr
import numpy as np
from PIL import Image
import cv2
from motion_brush_utils import MotionBrush

motion_brush = MotionBrush()

def reset_max_steps_to_replace_if_invalid(max_steps_to_replace, num_inference_steps):
    if max_steps_to_replace > num_inference_steps:
        return num_inference_steps
    else:
        return max_steps_to_replace

def resize_image_and_mask_if_invalid(image, mask, max_size=576*1024, length_factor=64):
    height, width = image.shape[:2]

    # Calculate the total size and compare with max_size
    if height * width > max_size:
        scale_factor = np.sqrt(max_size / (height * width))
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Adjust to ensure divisibility by length_factor
        new_height -= new_height % length_factor
        new_width -= new_width % length_factor

        # Resize with nearest neighbor interpolation
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        print("1: Resized image from {}x{} to {}x{}".format(height, width, new_height, new_width))

        height, width = new_height, new_width

    # Further resize if height or width is not divisible by length_factor
    if height % length_factor != 0 or width % length_factor != 0:
        new_height = height + - height % length_factor
        new_width = width + - width % length_factor

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        print("2: Resized image from {}x{} to {}x{}".format(height, width, new_height, new_width))

    return image, mask

def animate_image(
    image,
    num_inference_steps,
    max_steps_to_replace,
    num_frames,
    fps,
    motion_bucket_id,
    noise_aug,
    seed,
    max_height, max_width
):

    source_image = image['background'][..., :3]
    mask = image['layers'][0][..., -1:]
    mask = (mask > 128).astype(np.float32)

    source_image, mask = resize_image_and_mask_if_invalid(source_image, mask, max_size=int(max_height)*int(max_width))

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
        noise_aug_strength=float(noise_aug),
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
        with gr.Column(scale=2):
            num_inference_steps = gr.Slider(minimum=20, maximum=50, step=5, value=25, label="Number of Inference Steps")
            max_steps_to_replace = gr.Slider(minimum=1, maximum=50, step=1, value=25, label="Max Steps to Replace")
            motion_bucket_id = gr.Slider(minimum=0, maximum=300, step=1, value=30, label="Motion Bucket ID")
        with gr.Column(scale=6):
            with gr.Row():
                max_height = gr.Dropdown(label="Max Height", choices=[256, 512, 576, 768], value=256, allow_custom_value=True)
                max_width = gr.Dropdown(label="Max Width", choices=[256, 512, 576, 768, 1024], value=512, allow_custom_value=True)
            with gr.Row():
                num_frames = gr.Textbox(label="Number of Frames", value=25)
                fps = gr.Textbox(label="FPS", value=8)
                noise_aug = gr.Textbox(label="Noise Aug Strength", value=0.02)
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
            noise_aug,
            seed,
            max_height,
            max_width,
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