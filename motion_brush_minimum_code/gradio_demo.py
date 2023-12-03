import gradio as gr
import numpy as np
from motion_brush_utils import MotionBrush

motion_brush = MotionBrush()

def animate_image(image):

    source_image = image['background'][..., :3]
    mask = image['layers'][0][..., -1:]
    mask = (mask > 128).astype(np.float32)

    gif_path = motion_brush(source_image, mask)
    return gif_path

input_image = gr.ImageEditor(height=512)
output_image = gr.Image(height=512)

gr.Interface(fn=animate_image, inputs=input_image, outputs=output_image).launch(share=True)
