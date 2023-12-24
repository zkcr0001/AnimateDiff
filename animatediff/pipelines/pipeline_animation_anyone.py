# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet_2d_condition import UNet2DConditionModel
from ..models.animate_anyone_network_jinxu import UNet3DConditionModel
from ..models.animate_anyone_network_jinxu import PoseGuider3D
from ..models.animate_any_model_jinxu import AnimateAnyoneModel
from sentence_transformers import SentenceTransformer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationAnyonePipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationAnyonePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        referencenet: UNet2DConditionModel,
        poseguider3D: PoseGuider3D, 
        unet: UNet3DConditionModel,
        clip: SentenceTransformer,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            referencenet=referencenet,
            poseguider3D=poseguider3D, 
            unet=unet,
            clip=clip,
            scheduler=scheduler,
        )
        self.vae = vae
        self.clip = clip
        self.unet = unet
        self.poseguider3D = poseguider3D
        self.referencenet = referencenet
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.model = AnimateAnyoneModel(ReferenceNet=referencenet, Pose_Guider3D=poseguider3D, Unet_3D=unet)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        device = self.unet.device
        self.vae.to(device)
        self.clip.to(device)
        self.unet.to(device)
        self.poseguider3D.to(device)
        self.referencenet.to(device)
        return device

    def _encode_prompt(self, reference_image_pil, device):
        batch_size = len(reference_image_pil) if isinstance(reference_image_pil, list) else 1

        reference_prompt = self.clip.encode(reference_image_pil)
        prompt_embeddings = torch.from_numpy(reference_prompt).repeat(1, 77, 1).to(device)

        return prompt_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, openpose_pil_list, reference_image_pil, height, width, callback_steps):
        for img in openpose_pil_list:
            if height != img.size[1] or width != img.size[0]:
                raise ValueError(
                    f"Expected all images to have the same size as the `height` and `width` parameters but got"
                    f" {img.size[1]}x{img.size[0]} for image {img}."
                )
        if height != reference_image_pil.size[1] or width != reference_image_pil.size[0]:
            raise ValueError(
                f"Expected the reference image to have the same size as the `height` and `width` parameters but got"
                f" {reference_image_pil.size[1]}x{reference_image_pil.size[0]} for image {reference_image_pil}."
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        openpose_pil_list,
        reference_image_pil,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1

        device = self._execution_device

        # Check inputs. Raise error if not correct
        self.check_inputs(openpose_pil_list, reference_image_pil, height, width, callback_steps)
        video_length = len(openpose_pil_list)

        openpose_list = np.stack([np.array(img).astype(np.float32) / 255. for img in openpose_pil_list], axis=0)
        openpose_torch = torch.from_numpy(openpose_list).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        reference_image_torch = torch.from_numpy(np.array(reference_image_pil).astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).contiguous()

        openpose_torch = openpose_torch.repeat(num_videos_per_prompt, 1, 1, 1, 1).to(device)
        reference_image_torch = reference_image_torch.repeat(num_videos_per_prompt, 1, 1, 1).to(device)
        reference_image_pil = [reference_image_pil] * num_videos_per_prompt
        reference_latents = self.vae.encode(reference_image_torch).latent_dist.sample().to(device)
        reference_prompt = self.clip.encode(reference_image_pil, show_progress_bar=False)
        reference_embedding = torch.from_numpy(reference_prompt).unsqueeze(1).repeat(1, 77, 1).to(device)


        # Encode input prompt
        prompt_embeddings = self._encode_prompt(
            reference_image_pil, device
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeddings.dtype,
            device,
            generator,
        ) # (batch_size(1), latents_channel (4), frame_number(16), H // 8, W // 8)
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latents = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.model(latents, reference_latents, reference_embedding, openpose_torch, t).sample.to(dtype=latents_dtype)
                # noise_pred = []
                # import pdb
                # pdb.set_trace()
                # for batch_idx in range(latent_model_input.shape[0]):
                #     noise_pred_single = self.unet(latent_model_input[batch_idx:batch_idx+1], t, encoder_hidden_states=text_embeddings[batch_idx:batch_idx+1]).sample.to(dtype=latents_dtype)
                #     noise_pred.append(noise_pred_single)
                # noise_pred = torch.cat(noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationAnyonePipelineOutput(videos=video)

