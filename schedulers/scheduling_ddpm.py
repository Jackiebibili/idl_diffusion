from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from utils import randn_tensor


class DDPMScheduler(nn.Module):

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        """
        Args:
            num_train_timesteps (`int`):

        """
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.timesteps: torch.Tensor

        super(DDPMScheduler, self).__init__()

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # calculate betas
        if self.beta_schedule == "linear":
            # This is the DDPM implementation
            # note: betas are increasing linearly from beta_start (small) to beta_end (large) as the timestep increases
            betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=torch.float64,
            )
        else:
            raise NotImplementedError(f"Other beta schedule not implemented.")
        self.register_buffer("betas", betas)

        # calculate alphas
        # 1 - betas
        alphas = torch.ones_like(betas) - betas
        self.register_buffer("alphas", alphas)
        # calculate alpha cumulative product
        # cum prods i = product of the first i alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # timesteps
        # go from T-1 to 0
        timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1).long()
        self.register_buffer("timesteps", timesteps)
        self.timestep_to_index = {int(t.item()): i for i, t in enumerate(self.timesteps)}

    # use in inference to skip some timesteps for faster inference
    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,  # type: ignore
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # set timesteps
        timesteps = (
            torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                device=device,
            )
            .round()
            .long()
        )

        self.timesteps = timesteps
        self.timestep_to_index = {int(t.item()): i for i, t in enumerate(self.timesteps)}

    def __len__(self):
        return self.num_train_timesteps

    # used in inference to get the previous timestep for a given timestep
    def previous_timestep(self, timestep):
        """
        Get the previous timestep for a given timestep.

        Args:
            timestep (`int`): The current timestep.

        Return:
            prev_t (`int`): The previous timestep.
        """
        # length of steps
        num_inference_steps = (
            self.num_inference_steps
            if self.num_inference_steps
            else self.num_train_timesteps
        )

        # current index of timestep
        cur_ts_index = self.timestep_to_index[timestep]

        if cur_ts_index == num_inference_steps - 1:
            # if the current timestep is the last one (t=0), set previous timestep to -1
            prev_t = -1
        else:
            # calculate previous timestep
            prev_t = self.timesteps[cur_ts_index + 1]

        return prev_t

    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDPM. It calculates the variance $sigma_t$ for a given timestep.

        Args:
            t (`int`): The current timestep.

        Return:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """

        # calculate $beta_t$ for the current timestep using the cumulative product of alphas
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(
                1.0, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype
            )
        )
        current_beta_t = self.betas[t]

        # For t > 0, compute predicted variance $\beta_t$ (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        var = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        var = torch.clamp(var, min=1e-20)

        # we start with two types of variance as mentioned in Section 3.2 of https://arxiv.org/pdf/2006.11239.pdf
        # 1. fixed_small: $\sigma_t = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t, this one is optimal for $x_0$ being deterministic
        # 2. fixed_large: $\sigma_t^2 = \beta$, this one is optimal for $x_0 \sim mathcal{N}(0, 1)$
        if self.variance_type == "fixed_small":
            # fixed small variance
            variance = var
        elif self.variance_type == "fixed_large":
            # fixed large variance
            variance = current_beta_t
            # small hack: set the initial (log-)variance like so to get a better decoder log likelihood.
            variance = torch.clamp(variance, min=1e-20)
        else:
            raise NotImplementedError(
                f"Variance type {self.variance_type} not implemented."
            )

        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples. This function is used to add noise to the original samples at the beginning of each training iteration.


        Args:
            original_samples (`torch.Tensor`):
                The original samples.
            noise (`torch.Tensor`):
                The noise tensor.
            timesteps (`torch.IntTensor`):
                The timesteps.

        Return:
            noisy_samples (`torch.Tensor`):
                The noisy samples.
        """

        # make sure alphas the on the same device as samples
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)  # type: ignore

        # get sqrt alphas
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # get sqrt one miucs alphas
        one_minus_alpha_prod = torch.ones_like(alphas_cumprod) - alphas_cumprod
        sqrt_one_minus_alpha_prod = torch.sqrt(one_minus_alpha_prod)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # add noise to the original samples using the formula (14) from https://arxiv.org/pdf/2006.11239.pdf
        noisy_samples = (
            sqrt_alpha_prod[timesteps] * original_samples
            + sqrt_one_minus_alpha_prod[timesteps] * noise
        )
        return noisy_samples

    # used in inference to predict the previous sample from the current sample and model output
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """

        t = timestep
        prev_t = self.previous_timestep(t)

        # if prev_t == -1:
        #     return None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t].to(device=sample.device, dtype=sample.dtype)
            if prev_t >= 0
            else torch.tensor(1.0, device=sample.device, dtype=sample.dtype)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = self.alphas[t].to(device=sample.device, dtype=sample.dtype)
        current_beta_t = self.betas[t].to(device=sample.device, dtype=sample.dtype)

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (1 / torch.sqrt(alpha_prod_t)) * (
                sample - torch.sqrt(beta_prod_t) * model_output
            )
        else:
            raise NotImplementedError(
                f"Prediction type {self.prediction_type} not implemented."
            )

        # 3. Clip or threshold "predicted x_0" (for better sampling quality)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            torch.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
        )
        current_sample_coeff = (
            torch.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        if t > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            # use self.get_variance and variance_noise
            variance = self._get_variance(t).to(device=sample.device, dtype=sample.dtype)

            # add variance to prev_sample
            pred_prev_sample += torch.sqrt(variance) * variance_noise

        return pred_prev_sample
