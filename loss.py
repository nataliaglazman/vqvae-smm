"""Definition of loss functions."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from lpips import LPIPS
from torch import cat, reshape, tensor
from torch.fft import rfftn
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
        self.n_slices = 32  # slices per orientation (32×3 = 96 total, batched in one LPIPS call)
        self.perceptual_function = LPIPS(net="squeeze")
        # Freeze LPIPS — we never train it, so prevent PyTorch from
        # allocating gradient buffers for its parameters.  This saves
        # both memory and compute on every backward pass.
        self.perceptual_function.eval()
        for p in self.perceptual_function.parameters():
            p.requires_grad_(False)

        self.fft_factor = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def train(self, mode: bool = True):
        """Override to keep LPIPS permanently in eval mode."""
        super().train(mode)
        self.perceptual_function.eval()
        return self

    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements
        x = y.float()
        y = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

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

    def _calculate_frequency_loss(self, x, y, max_voxels: int = 128**3) -> torch.Tensor:
        # Compute FFT on the batch.  For large 3D volumes the full-resolution
        # rfftn and its backward graph consume several GB of GPU memory.
        # When the volume exceeds *max_voxels*, we downsample to a manageable
        # size first — the frequency loss is still meaningful at lower
        # resolution and the memory saving is dramatic.
        with torch.cuda.amp.autocast(enabled=False):
            # fftn requires float32; x/y may be float16 under AMP
            x_f = (x.float() + 1.0) / 2.0
            y_f = (y.float() + 1.0) / 2.0

            n_voxels = x_f[0, 0].numel()
            if n_voxels > max_voxels:
                scale = (max_voxels / n_voxels) ** (1.0 / 3.0)
                target = [max(1, int(s * scale)) for s in x_f.shape[2:]]
                x_f = F.interpolate(x_f, size=target, mode="trilinear", align_corners=False)
                y_f = F.interpolate(y_f, size=target, mode="trilinear", align_corners=False)

            loss = F.mse_loss(torch.abs(rfftn(x_f, norm="ortho")), torch.abs(rfftn(y_f, norm="ortho"))).to(x.dtype)

        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss

        return loss

    def _calculate_pixel_loss(self, x, y) -> torch.Tensor:
        loss = F.l1_loss(x, y)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        # Batched perceptual loss across 3 orientations.
        #
        # Collects random slices from sagittal, coronal and axial planes,
        # resizes them to a common spatial size, then evaluates them in a
        # *single* LPIPS forward pass.  Compared to the previous 3-call
        # approach with 128 slices each, this is ~4× faster and uses ~4×
        # less backprop memory.
        #
        # Gradient note: x (ground truth) is detached so only y
        # (reconstruction) carries gradients back to the decoder.

        common_size = (96, 96)
        all_x, all_y = [], []
        counts = []  # slices per orientation, for splitting later

        for perm in [
            (0, 2, 1, 3, 4),  # sagittal  – slice along D
            (0, 4, 1, 2, 3),  # axial     – slice along W
            (0, 3, 1, 2, 4),  # coronal   – slice along H
        ]:
            x_p = x.permute(*perm)  # (B, N_total, C, H', W')
            y_p = y.permute(*perm)
            n_total = x_p.shape[1]
            n_sel = min(self.n_slices, n_total)
            idx = torch.randperm(n_total, device=x.device)[:n_sel]

            sx = x_p[:, idx].flatten(0, 1)  # (B*n_sel, C, h, w)
            sy = y_p[:, idx].flatten(0, 1)

            # Resize to common dims so we can cat across orientations
            sx = F.interpolate(sx, size=common_size, mode="bilinear", align_corners=False).detach()
            sy = F.interpolate(sy, size=common_size, mode="bilinear", align_corners=False)

            all_x.append(sx)
            all_y.append(sy)
            counts.append(sx.shape[0])

        # ── Single LPIPS forward pass ────────────────────────────────
        cat_x = torch.cat(all_x, dim=0).float()
        cat_y = torch.cat(all_y, dim=0).float()
        per_slice = self.perceptual_function.forward(cat_x, cat_y).view(-1)

        # Split back for per-orientation logging
        p_sag, p_ax, p_cor = [s.mean() for s in per_slice.split(counts)]

        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Sagittal-Reconstruction"] = p_sag
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Axial-Reconstruction"] = p_ax
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Coronal-Reconstruction"] = p_cor

        loss = (p_sag + p_ax + p_cor) * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
