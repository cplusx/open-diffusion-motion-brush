# Minimum Motion Brush Code Repository
This repository contains the minimum code to implement the motion brush using the diffusers and gradio.

## How to use

To launch the Gradio demo, simply run the following command in your terminal:
```bash
python gradio_demo.py
```

## Implementation Details of the Motion Brush
### Overriding the Scheduler Sampler
The key code is the overridden class of the scheduler sampler in [motion_brush_utils.py](motion_brush_utils.py)

The class is named `EulerDiscreteSchedulerMotionBrush`, it is inherited from `EulerDiscreteScheduler` in [diffusers](https://github.com/huggingface/diffusers).

In this inherited class, it overrides the `step` function with one additional line of code. 
```python
pred_original_sample = self.replace_prediction_with_mask(pred_original_sample, self.mask)
```
Aside from this addition, the rest of the `step`` function remains identical to the original `EulerDiscreteScheduler`` class.


## The Key Function: replace_prediction_with_mask
The function `EulerDiscreteSchedulerMotionBrush.replace_prediction_with_mask` plays a pivotal role in achieving the motion brush. The code snippet is as follows:

```python
def replace_prediction_with_mask(self, prediction, mask):
    '''
    for frames from 2 to end, replace the region where mask == 0 with the prediction of frame 1
    '''
    if mask is None:
        return prediction
    *_, height, width = prediction.shape
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(prediction.device)
    mask = mask.squeeze()
    while mask.dim() < prediction.dim() - 1:
        mask = mask.unsqueeze(0)
    resized_mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="bilinear").unsqueeze(0)

    prediction[:, 1:] = torch.where(resized_mask > 0.5, prediction[:, 1:], prediction[:, 0:1])
    return prediction
```


* Mask: We use a binary mask where 1 indicates regions to add motion, and 0 represents areas to remain static. This binary mask is obtained from the user's brush strokes in the Gradio interface.
* Replacement Strategy: The prediction for frames 2 onwards is replaced with the prediction from the first frame. The intuition is that when the Stable Video Diffusion model makes animation, the first transferred frame is quite similar to the condition image, so we can use the first frame to replace the prediction of the subsequent frames to obtain a static region across the video. While for the region where the mask is 1, we keep the prediction from the model.
* Result: The static regions maintain consistency with the condition image, while the masked regions exhibit motion as predicted by the model.

That is all the idea to achieve the motion brush with Stabilityai's [Stable Video Diffusion](https://huggingface.co/docs/diffusers/main/using-diffusers/svd). 