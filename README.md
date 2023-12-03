# open-diffusion-motion-brush
An Open Implementation of Motion Brush like Gen-2

Work in progress...🚧

For those who want to try simple demo or want to figure out my trick to make the motion brush, I put the minimum code under the repository [motion_brush_minimum_code](motion_brush_minimum_code). It is quite simple and easy to understand.

## Installation
First install the [PyTorch](https://pytorch.org/get-started/locally/) according to your device.
```bash
git clone https://github.com/cplusx/open-diffusion-motion-brush.git
pip install -r requirements.txt
```

## How to use
Start the demo by running:
```bash
python gradio_demo.py
```

The following GIF shows how to use the motion brush to specify the region to have motion. 

(Note, the GIF removes the processing frames, it takes ~2 mins on a V100 for 25x576x1024 video.)
![figures/motion_brush_demo.gif](figures/motion_brush_demo.gif)

Functions TODO list
- [] Mask dilation
- [x] Set number of time steps to do replacement