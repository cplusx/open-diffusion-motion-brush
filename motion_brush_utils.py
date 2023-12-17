import torch
import numpy as np
from typing import Any, Optional, Tuple, Union
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler, EulerDiscreteSchedulerOutput, logger
from diffusers.utils.torch_utils import randn_tensor
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from image_utils import make_gif

class EulerDiscreteSchedulerMotionBrush(EulerDiscreteScheduler):
    def __init__(self, *args, mask=None, max_steps_to_replace=None, **kwargs):
        '''
        mask: np.ndarray or torch.Tensor, shape (height, width) or (1xN, height, width), dtype float32, range [0, 1]
        max_steps_to_replace: int, the maximum number of timesteps to replace with the first frame. This number should be smaller than the number of timesteps in the diffusion chain. If None, all timesteps will be replaced.
        '''
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.max_steps_to_replace = max_steps_to_replace

    def set_motion_brush_arguments(
        self, 
        mask=None, 
        max_steps_to_replace=None
    ):
        self.mask = mask
        self.max_steps_to_replace = max_steps_to_replace

    def replace_prediction_with_mask(
        self, 
        prediction, 
    ):
        '''
        for frames from 2 to end, replace the region where mask == 0 with the prediction of frame 1
        '''
        mask = self.mask
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

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        
        if self.max_steps_to_replace is not None and self.step_index < self.max_steps_to_replace:
            pred_original_sample = self.replace_prediction_with_mask(pred_original_sample)

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

class MotionBrush():
    def __init__(self):
        self.pipe = None

    def _init_pipe(self):
        if self.pipe is None:
            scheduler = EulerDiscreteSchedulerMotionBrush.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder='scheduler')
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16",
                scheduler=scheduler
            )
            pipe.enable_model_cpu_offload()
            self.pipe = pipe

    def __call__(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        max_steps_to_replace: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        fps: Optional[int] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if self.pipe is None:
            self._init_pipe()

        # compute a unique name with hash for image
        image_name = str(hash(image.tobytes()))

        assert isinstance(self.pipe.scheduler, EulerDiscreteSchedulerMotionBrush)
        self.pipe.scheduler.set_motion_brush_arguments(
            mask=mask, 
            max_steps_to_replace=max_steps_to_replace,
        )

        if (seed is not None) and (seed >= 0):
            generator = torch.manual_seed(seed)
        else:
            generator = None

        with torch.cuda.amp.autocast(dtype=torch.float16):
            frames = self.pipe(
                Image.fromarray(image),
                decode_chunk_size=5, 
                generator=generator, 
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
            ).frames
        frames_np = [np.array(frame) for frame in frames[0]]
        make_gif(frames_np, f"tmp/{image_name}.gif", fps=fps, rescale=0.5)
        return f'tmp/{image_name}.gif'