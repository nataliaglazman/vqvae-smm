"""Definition of loss functions."""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from torch import cat, reshape, tensor
from torch.fft import fftn
from torch.nn import PairwiseDistance

from utils import TBSummaryTypes

class BaurLoss(object):
    def __init__(self, lambda_reconstruction=1):
        super(BaurLoss).__init__()

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gdl = 0

        # Use mean instead of sum for proper scaling with 3D images
        self.l1_loss = lambda x, y: PairwiseDistance(p=1)(x.view(x.shape[0], -1), y.view(y.shape[0], -1)).mean()
        self.l2_loss = lambda x, y: PairwiseDistance(p=2)(x.view(x.shape[0], -1), y.view(y.shape[0], -1)).mean()

    def __call__(self, originals, reconstructions):
        summaries = {}

        l1_reconstruction = self.l1_loss(originals, reconstructions) * self.lambda_reconstruction
        l2_reconstruction = self.l2_loss(originals, reconstructions) * self.lambda_reconstruction

        summaries[("summaries", "scalar", "L1-Reconstruction-Loss")] = l1_reconstruction.item()
        summaries[("summaries", "scalar", "L2-Reconstruction-Loss")] = l2_reconstruction.item()
        summaries[("summaries", "scalar", "Lambda-Reconstruction")] = self.lambda_reconstruction

        originals_gradients = self.__image_gradients(originals)
        reconstructions_gradients = self.__image_gradients(reconstructions)

        l1_gdl = (
            self.l1_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l1_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l1_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        l2_gdl = (
            self.l2_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l2_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l2_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        summaries[("summaries", "scalar", "L1-Image_Gradient-Loss")] = l1_gdl.item()
        summaries[("summaries", "scalar", "L2-Image_Gradient-Loss")] = l2_gdl.item()
        summaries[("summaries", "scalar", "Lambda-Image_Gradient")] = self.lambda_gdl

        loss_total = l1_reconstruction + l2_reconstruction + l1_gdl + l2_gdl

        summaries[("summaries", "scalar", "Total_Loss")] = loss_total.item()

        return loss_total, summaries

    def set_lambda_reconstruction(self, lambda_reconstruction):
        self.lambda_reconstruction = lambda_reconstruction
        return self.lambda_reconstruction

    def set_lambda_gdl(self, lambda_gdl):
        self.lambda_gdl = lambda_gdl
        return self.lambda_gdl

    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, features, depth, height, width = input_shape

        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        dzz = tensor(()).new_zeros(
            (batch_size, features, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = cat([dz, dzz], 2)
        dz = reshape(dz, input_shape)

        dyz = tensor(()).new_zeros((batch_size, features, depth, 1, width), device=image.device, dtype=dy.dtype)
        dy = cat([dy, dyz], 3)
        dy = reshape(dy, input_shape)

        dxz = tensor(()).new_zeros(
            (batch_size, features, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = cat([dx, dxz], 4)
        dx = reshape(dx, input_shape)

        return dx, dy, dz


class BaselineLoss(torch.nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

        self.pixel_factor = 1.0

        self.perceptual_factor = 0.002
        self.n_slices = 128  # Reduced from 512 to save GPU memory
        self.perceptual_function = LPIPS(net="squeeze")

        self.fft_factor = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements
        x = y.float()
        y = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        print(f"Reconstruction tensor shape: {y.shape}")
        print(f"Quantization losses: {[q_loss.item() for q_loss in q_losses]}")
        print(f"Original image tensor shape: {x.shape}")

        loss = (
            self._calculate_pixel_loss(x, y)
            + self._calculate_frequency_loss(x, y)
            + self._calculate_perceptual_loss(x, y)
        )

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_frequency_loss(self, x, y) -> torch.Tensor:
        # Compute FFT on the full batch at once — cheaper than a per-sample loop
        # that builds a long autograd chain. Complex tensors are freed immediately
        # after mse_loss since we don't store them.
        with torch.cuda.amp.autocast(enabled=False):
            # fftn requires float32; x/y may be float16 under AMP
            x_f = (x.float() + 1.0) / 2.0
            y_f = (y.float() + 1.0) / 2.0
            loss = F.mse_loss(torch.abs(fftn(x_f, norm="ortho")), torch.abs(fftn(y_f, norm="ortho"))).to(x.dtype)

        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss

        return loss

    def _calculate_pixel_loss(self, x, y) -> torch.Tensor:
        loss = F.l1_loss(x, y)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        # LPIPS backbone weights are frozen (requires_grad=False), so autograd
        # will not accumulate gradients into the backbone parameters.  However,
        # gradients DO need to flow back through the LPIPS computation to the
        # reconstruction tensor `y` so that the decoder is trained by the
        # perceptual objective.
        #
        # Previously this was wrapped in torch.no_grad() + .detach(), which
        # made the perceptual loss completely non-trainable (zero gradient to
        # the decoder).  The wrapper and detach have been removed to restore
        # the correct gradient path: LPIPS → sel_y → decoder.
        #
        # Memory note: SqueezeNet activations for `self.n_slices` (128) slices
        # per orientation are retained for backprop.  With perceptual_factor=0.002
        # this is acceptable; reduce n_slices if memory is tight.

        def _lpips_on_slices(x_vol, y_vol, perm_dims):
            """Extract 2D slices along one orientation and compute LPIPS."""
            # Permute so the slice axis is dim=1: (B, n_slices_total, C, H, W)
            # Then index along dim=1 BEFORE flattening, so we never materialise
            # the full (B*n_slices_total, C, H, W) intermediate tensor.
            x_p = x_vol.permute(*perm_dims)  # (B, n_slices_total, C, H, W)
            n_slices_total = x_p.shape[1]
            indices = torch.randperm(n_slices_total, device=x_vol.device)[: self.n_slices]
            # (B, self.n_slices, C, H, W) -> (B * self.n_slices, C, H, W)
            # x (ground truth) does not require grad; detach to avoid storing
            # its graph.  y (reconstruction) keeps its computation graph intact.
            sel_x = x_p[:, indices].contiguous().flatten(0, 1).detach()
            del x_p
            sel_y = y_vol.permute(*perm_dims)[:, indices].contiguous().flatten(0, 1)
            p_loss = torch.mean(self.perceptual_function.forward(sel_x.float(), sel_y.float()))
            return p_loss

        # Sagittal
        p_loss_sagital = _lpips_on_slices(x, y, perm_dims=(0, 2, 1, 3, 4))
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Sagittal-Reconstruction"] = p_loss_sagital

        # Axial
        p_loss_axial = _lpips_on_slices(x, y, perm_dims=(0, 4, 1, 2, 3))
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Axial-Reconstruction"] = p_loss_axial

        # Coronal
        p_loss_coronal = _lpips_on_slices(x, y, perm_dims=(0, 3, 1, 2, 4))
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Coronal-Reconstruction"] = p_loss_coronal

        loss = (p_loss_sagital + p_loss_axial + p_loss_coronal) * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
