import os
from PIL import Image
import numpy as np
import imageio

def make_gif(frames, filename, fps=8, rescale=0.5):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # resize frames
    if rescale is not None:
        frames = [Image.fromarray(frame) for frame in frames]
        frames = [frame.resize((int(frame.width * rescale), int(frame.height * rescale))) for frame in frames]
        frames = [np.array(frame) for frame in frames]
    imageio.mimsave(filename, frames, duration=1000 / fps, loop=0)