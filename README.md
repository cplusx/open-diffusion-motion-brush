# open-diffusion-motion-brush
This Is An Open Implementation of Motion Brush like Gen-2
Motion Brush allows you to specify the region of image to have motion.
This implementation is based on 
* [diffusers](https://github.com/huggingface/diffusers)
* [Stable Video Diffusion](https://huggingface.co/docs/diffusers/main/using-diffusers/svd) by Stabilityai

🔥🚀 Try it in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/cplusx/4fbe79162880cd007cdca39389b3c4a4/open_motion_brush.ipynb)

## Introduction
Steps to have your image to be moving/animated:
1. Upload your image
2. Use the sketch brush to specify the region to have motion
3. [Optinoal] Adjust the parameters, the motion bucket ID will affect the motion strength (the larger the stronger)
4. Click "Generate" to see the result

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
- [] Google Colab Demo
- [x] Handle image resizing if the input image is too large or does not satisfy the requirement
- [x] Set number of time steps to do replacement