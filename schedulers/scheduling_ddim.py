from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from utils import randn_tensor

from .scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.num_inference_steps is not None
        ), "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)

    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDIM. It calculates the variance $sigma_t$ for a given timestep.

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
            else torch.tensor(1.0, device=alpha_prod_t.device, dtype=alpha_prod_t.dtype)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # DDIM equation for variance
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        # sd at timestep t is later computed: eta * sqrt(variance)

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float = 0.0,
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
            eta (`float`):
                The weight of the noise to add to the variance.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        t = timestep
        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=sample.device, dtype=sample.dtype)
        )  # alpha_prod_t_prev is 1 (no noise added) if prev_t == -1
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - torch.sqrt(beta_prod_t) * model_output
            ) / torch.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        elif self.prediction_type == "v_prediction":
            # x0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
            pred_original_sample = (
                torch.sqrt(alpha_prod_t) * sample
                - torch.sqrt(beta_prod_t) * model_output
            )

            # eps = sqrt(alpha_bar_t) * v + sqrt(1 - alpha_bar_t) * x_t
            pred_epsilon = (
                torch.sqrt(alpha_prod_t) * model_output
                + torch.sqrt(beta_prod_t) * sample
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

        # 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(t)
        std_dev_t = eta * torch.sqrt(variance)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (
            torch.sqrt(torch.clamp(beta_prod_t_prev - std_dev_t**2, min=1e-20))
            * pred_epsilon
        )

        # 6. compute x_t-1 without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # 7. Add noise with eta
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample += std_dev_t * variance_noise

        return prev_sample
