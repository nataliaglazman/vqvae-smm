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


class BaselineLossOLD(torch.nn.Module):
    def __init__(self, commitment_weight: float = 0.25):
        super(BaselineLossOLD, self).__init__()

        self.pixel_factor = 1.0
        self.gdl_factor = 1.0

        self.perceptual_factor = 1.0
        self.n_slices = 32  # slices per orientation (32×3 = 96 total, batched in one LPIPS call)
        self.perceptual_function = LPIPS(net="squeeze")
        # Freeze LPIPS — we never train it, so prevent PyTorch from
        # allocating gradient buffers for its parameters.  This saves
        # both memory and compute on every backward pass.
        self.perceptual_function.eval()
        for p in self.perceptual_function.parameters():
            p.requires_grad_(False)

        self.fft_factor = 10.0
        self.commitment_weight = commitment_weight

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def train(self, mode: bool = True):
        """Override to keep LPIPS permanently in eval mode."""
        super().train(mode)
        self.perceptual_function.eval()
        return self

    def forward(self, network_output: Dict[str, List[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        # gt = ground truth, recon = reconstruction
        gt = target.float()
        recon = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        loss = (
            self._calculate_pixel_loss(gt, recon)
            + self._calculate_frequency_loss(gt, recon)
            + self._calculate_perceptual_loss(gt, recon)
            + self._calculate_gdl(gt, recon)
        )

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss

            loss = loss + q_loss * self.commitment_weight

        return loss

    def _calculate_frequency_loss(self, gt, recon, max_voxels: int = 128**3) -> torch.Tensor:
        # Compute FFT on the batch.  For large 3D volumes the full-resolution
        # rfftn and its backward graph consume several GB of GPU memory.
        # When the volume exceeds *max_voxels*, we downsample to a manageable
        # size first — the frequency loss is still meaningful at lower
        # resolution and the memory saving is dramatic.
        with torch.amp.autocast("cuda", enabled=False):
            # fftn requires float32; gt/recon may be float16 under AMP.
            # Data is zero-mean/unit-std from NormalizeIntensityd, so we
            # normalise to [0, 1] using the actual batch range (not a
            # hard-coded [-1, 1] assumption).
            gt_f = gt.float()
            recon_f = recon.float()
            batch_min = gt_f.amin(dim=(1, 2, 3, 4), keepdim=True)
            batch_max = gt_f.amax(dim=(1, 2, 3, 4), keepdim=True)
            denom = (batch_max - batch_min).clamp(min=1e-8)
            gt_f = (gt_f - batch_min) / denom
            recon_f = (recon_f - batch_min) / denom  # same scale as gt

            n_voxels = gt_f[0, 0].numel()
            if n_voxels > max_voxels:
                scale = (max_voxels / n_voxels) ** (1.0 / 3.0)
                target = [max(1, int(s * scale)) for s in gt_f.shape[2:]]
                gt_f = F.interpolate(gt_f, size=target, mode="trilinear", align_corners=False)
                recon_f = F.interpolate(recon_f, size=target, mode="trilinear", align_corners=False)

            loss = F.mse_loss(torch.abs(rfftn(gt_f, norm="ortho")), torch.abs(rfftn(recon_f, norm="ortho"))).to(gt.dtype)

        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss

        return loss

    def _calculate_pixel_loss(self, gt, recon) -> torch.Tensor:
        loss = F.l1_loss(gt, recon)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss

        return loss

    def _calculate_perceptual_loss(self, gt, recon) -> torch.Tensor:
        # Batched perceptual loss across 3 orientations.
        #
        # Collects random slices from sagittal, coronal and axial planes,
        # resizes them to a common spatial size, then evaluates them in a
        # *single* LPIPS forward pass.  Compared to the previous 3-call
        # approach with 128 slices each, this is ~4× faster and uses ~4×
        # less backprop memory.
        #
        # Gradient note: gt (ground truth) is detached so only recon
        # (reconstruction) carries gradients back to the decoder.

        common_size = (96, 96)
        all_gt, all_recon = [], []
        counts = []  # slices per orientation, for splitting later

        for perm in [
            (0, 2, 1, 3, 4),  # sagittal  – slice along D
            (0, 4, 1, 2, 3),  # axial     – slice along W
            (0, 3, 1, 2, 4),  # coronal   – slice along H
        ]:
            gt_p = gt.permute(*perm)  # (B, N_total, C, H', W')
            recon_p = recon.permute(*perm)
            n_total = gt_p.shape[1]
            n_sel = min(self.n_slices, n_total)
            idx = torch.randperm(n_total, device=gt.device)[:n_sel]

            s_gt = gt_p[:, idx].flatten(0, 1)  # (B*n_sel, C, h, w)
            s_recon = recon_p[:, idx].flatten(0, 1)

            # Resize to common dims so we can cat across orientations
            s_gt = F.interpolate(s_gt, size=common_size, mode="bilinear", align_corners=False).detach()
            s_recon = F.interpolate(s_recon, size=common_size, mode="bilinear", align_corners=False)

            all_gt.append(s_gt)
            all_recon.append(s_recon)
            counts.append(s_gt.shape[0])

        # ── Single LPIPS forward pass ────────────────────────────────
        # LPIPS expects inputs in [-1, 1].  Data is zero-mean/unit-std from
        # NormalizeIntensityd so we clamp to [-1, 1] to match the pretrained
        # backbone's expected range.
        cat_gt = torch.cat(all_gt, dim=0).float().clamp(-1, 1)
        cat_recon = torch.cat(all_recon, dim=0).float().clamp(-1, 1)
        per_slice = self.perceptual_function.forward(cat_gt, cat_recon).view(-1)

        # Split back for per-orientation logging
        p_sag, p_ax, p_cor = [s.mean() for s in per_slice.split(counts)]

        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Sagittal-Reconstruction"] = p_sag
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Axial-Reconstruction"] = p_ax
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Coronal-Reconstruction"] = p_cor

        loss = (p_sag + p_ax + p_cor) * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss

        return loss

    def _calculate_gdl(self, gt, recon) -> torch.Tensor:
        """Gradient Domain Loss — penalizes differences in spatial gradients to sharpen edges."""
        dx_gt = gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1]
        dy_gt = gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :]
        dz_gt = gt[:, :, 1:, :, :] - gt[:, :, :-1, :, :]

        dx_recon = recon[:, :, :, :, 1:] - recon[:, :, :, :, :-1]
        dy_recon = recon[:, :, :, 1:, :] - recon[:, :, :, :-1, :]
        dz_recon = recon[:, :, 1:, :, :] - recon[:, :, :-1, :, :]

        loss = (
            F.l1_loss(dx_gt, dx_recon)
            + F.l1_loss(dy_gt, dy_recon)
            + F.l1_loss(dz_gt, dz_recon)
        ) * self.gdl_factor

        self.summaries[TBSummaryTypes.SCALAR]["Loss-GDL-Reconstruction"] = loss
        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class BaselineLoss(torch.nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

        self.pixel_factor = 1.0

        self.perceptual_factor = 0.002
        self.n_slices = 16
        self.perceptual_function = LPIPS(net="alex")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements — compute entirely in float32 to avoid
        # float16 overflow (FFT magnitudes and perceptual scaling can
        # exceed the float16 range after enough training steps).
        x = y.float()
        y = network_output["reconstruction"][0].float()

        # The decoder has no output activation, so y can have arbitrarily
        # large values.  Clamp to the expected input range [-1, 1] to
        # prevent the FFT magnitude and LPIPS from amplifying outliers
        # into NaN.  Gradients still flow through non-clamped voxels;
        # the clamp only kills gradient for values already far outside
        # the valid range (desired — push them back via the pixel loss,
        # not via an exploding FFT gradient).
        y = y.clamp(-1.0, 1.0)

        q_losses = network_output["quantization_losses"]

        mask = network_output.get("mask")
        if mask is not None:
            mask = mask.float()
            # Zero reconstruction outside the brain so the FFT / perceptual
            # terms see the same support as the pixel term.
            y = y * mask

        loss = self._calculate_pixel_loss(x, y, mask=mask) + self._calculate_perceptual_loss(x, y)

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss.detach()

            loss = loss + q_loss

        return loss

    def _calculate_pixel_loss(self, x, y, mask=None) -> torch.Tensor:
        if mask is None:
            loss = F.l1_loss(x, y)
        else:
            # Mean over brain voxels only — avoids diluting the loss with
            # background zeros whose target and prediction are both ~0.
            diff = (x - y).abs() * mask
            denom = mask.sum().clamp_min(1.0)
            loss = diff.sum() / denom
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss.detach()

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        def _lpips_on_slices(x_vol, y_vol, perm_dims):
            x_p = x_vol.permute(*perm_dims)
            n_slices_total = x_p.shape[1]
            indices = torch.randperm(n_slices_total, device=x_vol.device)[: self.n_slices]
            sel_x = x_p[:, indices].contiguous().flatten(0, 1).detach()
            del x_p
            sel_y = y_vol.permute(*perm_dims)[:, indices].contiguous().flatten(0, 1)
            if sel_x.shape[-1] > 96 or sel_x.shape[-2] > 96:
                _target = (min(sel_x.shape[-2], 96), min(sel_x.shape[-1], 96))
                sel_x = F.adaptive_avg_pool2d(sel_x, _target)
                sel_y = F.adaptive_avg_pool2d(sel_y, _target)
            if sel_x.shape[1] == 1:
                sel_x = sel_x.expand(-1, 3, -1, -1)
                sel_y = sel_y.expand(-1, 3, -1, -1)
            p_loss = torch.mean(self.perceptual_function.forward(sel_x.float(), sel_y.float()))
            if not torch.isfinite(p_loss):
                return torch.zeros(1, device=x_vol.device, dtype=torch.float32, requires_grad=True).squeeze()
            return p_loss

        orientations = [
            ("Sagittal", (0, 2, 1, 3, 4)),
            ("Axial", (0, 4, 1, 2, 3)),
            ("Coronal", (0, 3, 1, 2, 4)),
        ]

        total_p_loss = torch.zeros(1, device=x.device, dtype=torch.float32)
        for name, perm_dims in orientations:
            p_loss = _lpips_on_slices(x, y, perm_dims=perm_dims)
            self.summaries[TBSummaryTypes.SCALAR][f"Loss-Perceptual_{name}-Reconstruction"] = p_loss.detach()
            total_p_loss = total_p_loss + p_loss

        loss = total_p_loss * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss.detach()

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries



class CliffLoss(torch.nn.Module):
    """Cliff disentanglement loss (Barin-Pacela et al., 2024).

    Encourages axis-aligned discontinuities (cliffs) in the learned latent
    density via three terms:
      - l_uni:    minimise entropy of gradient-magnitude density per marginal
      - l_biv:    minimise JSD of conditional gradient-magnitude densities
      - l_KL-uni: KL(Uniform || marginal) to prevent collapse to Diracs

    Includes a learnable **nonlinear** projection (MLP) from raw encoder
    channels down to ``latent_dim`` factors so that the projected factors
    can develop non-Gaussian marginal shapes (cliffs).  A purely linear
    projection preserves Gaussianity of pooled features, making the
    normalised gradient-magnitude entropy a constant.
    """

    def __init__(
        self,
        lambda_uni: float = 1.0,
        lambda_biv: float = 1.0,
        lambda_kl_uni: float = 1.0,
        sigma: float | None = None,
        K: int = 100,
        M: int = 10,
        latent_dim: int = 32,
        in_dim: int | None = None,
        z_min: float = -5.0,
        z_max: float = 5.0,
    ):
        super(CliffLoss, self).__init__()
        self.lambda_uni = lambda_uni
        self.lambda_biv = lambda_biv
        self.lambda_kl_uni = lambda_kl_uni
        self._sigma_override = sigma  # None → auto via Silverman's rule
        self.K = K
        self.M = M
        self.latent_dim = latent_dim
        self.z_min = z_min
        self.z_max = z_max
        # Learnable nonlinear projection: raw encoder channels → latent factors
        # An MLP (rather than a single linear layer) allows the projected
        # factors to take non-Gaussian shapes, which is essential for the
        # entropy-based uni/biv terms to produce a useful gradient signal.
        if in_dim is not None:
            self.proj = self._build_proj(in_dim, latent_dim)
        else:
            self.proj = None  # will be lazily created on first forward
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_proj(in_features: int, latent_dim: int) -> torch.nn.Sequential:
        """Build a small MLP projection: Linear → GELU → Linear."""
        hidden = max(in_features // 2, latent_dim * 2)
        proj = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, latent_dim),
        )
        # Initialise for stable gradients at start
        for m in proj:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        return proj

    def _ensure_proj(self, in_features: int, device: torch.device):
        """Create the projection MLP on first call (lazy init)."""
        if self.proj is None:
            self.proj = self._build_proj(in_features, self.latent_dim).to(device)

    def _bandwidth(self, n: int) -> float:
        """Return KDE bandwidth.  If user set sigma explicitly, use that.
        Otherwise apply Silverman's rule: σ = n^(−1/5) (data is already
        standardised to unit variance)."""
        if self._sigma_override is not None:
            return self._sigma_override
        return max(n ** (-0.2), 0.1)  # floor at 0.1 for very large batches

    @staticmethod
    def _standardize(z: torch.Tensor) -> torch.Tensor:
        """Standardize each factor to zero mean and unit variance (per-batch)."""
        mu = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (z - mu) / std

    def _kde_kernels(self, z: torch.Tensor, grid: torch.Tensor, sigma: float):
        """Precompute Gaussian KDE kernels and their derivatives for all dims.

        Args:
            z: (n, d) standardised latent factors
            grid: (K,) evaluation points
            sigma: KDE bandwidth

        Returns:
            kernel:  (d, K, n) — Gaussian kernel values
            dkernel: (d, K, n) — derivative of kernel w.r.t. grid position
        """
        # z.T → (d, n);  grid → (K,)
        # diff: (d, K, n) = grid[k] − z[l, i]
        diff = grid.view(1, -1, 1) - z.T.unsqueeze(1)
        kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (
            sigma * (2 * torch.pi) ** 0.5
        )
        dkernel = (-diff / sigma ** 2) * kernel
        return kernel, dkernel

    # ── sub-losses (fully vectorized) ─────────────────────────────────────

    def univariate_cliff_loss(self, z: torch.Tensor, grid: torch.Tensor, dz: float,
                              dkernel: torch.Tensor) -> torch.Tensor:
        """l_uni = Σ_i H(s_i) — entropy of normalised gradient magnitude of each marginal."""
        # dkernel: (d, K, n)
        dp_dz = dkernel.mean(dim=2)          # (d, K)
        abs_grad = dp_dz.abs()
        c = abs_grad.sum(dim=1, keepdim=True) * dz  # (d, 1)
        si = abs_grad / (c + 1e-12)          # (d, K)
        H = -(si * torch.log(si + 1e-12)).sum(dim=1) * dz  # (d,)
        return H.sum()

    def bivariate_cliff_loss(self, z: torch.Tensor, grid: torch.Tensor, dz: float,
                             kernel: torch.Tensor, dkernel: torch.Tensor,
                             sigma: float) -> torch.Tensor:
        """l_biv = Σ_{i≠j} JSD of conditional gradient-magnitude densities.

        Vectorised over all (i, j) pairs by chunking over the conditioning
        dimension j to keep peak memory bounded.
        """
        n, d = z.shape
        M = min(self.M, n)
        K = grid.shape[0]

        # Shared sample indices for conditioning values
        indices = torch.randperm(n, device=z.device)[:M]

        # Precomputed: dkernel (d, K, n) — derivative for the i dimension
        # We need kernel_j for each j at the M conditioning points.
        # z_xi: (M, d) — sampled conditioning vectors
        z_xi = z[indices]  # (M, d)

        # diff_j for all j: z_xi[:, j] − z[:, j]  →  (d, M, n)
        diff_j_all = z_xi.T.unsqueeze(2) - z.T.unsqueeze(1)  # (d, M, n)
        kernel_j_all = torch.exp(-0.5 * (diff_j_all / sigma) ** 2) / (
            sigma * (2 * torch.pi) ** 0.5
        )  # (d, M, n)

        # p(z_j = ξ_k) marginal via 1D KDE: (d, M)
        p_zj_all = kernel_j_all.mean(dim=2)  # (d, M)

        # Precompute per-j masks to zero out the i==j diagonal entry.
        # Each mask is a fresh tensor — no in-place mutation.
        eye = torch.eye(d, device=z.device)  # (d, d)

        # --- Compute JSD for all (i, j) pairs, chunking over j ---
        total_jsd = torch.tensor(0.0, device=z.device)

        # dkernel_i transposed for matmul: (d, n, K)
        dkernel_T = dkernel.permute(0, 2, 1)  # (d, n, K)

        for j in range(d):
            # kernel_j: (M, n) — conditioning kernel for dimension j
            kj = kernel_j_all[j]  # (M, n)
            pj = p_zj_all[j]     # (M,)

            # Joint derivative for ALL i dims at once:
            # dp_joint[i] = kj @ dkernel_T[i] / n  →  batch matmul
            # kj: (M, n),  dkernel_T: (d, n, K)
            # → (d, M, K) via einsum
            dp_joint = torch.einsum("mn,ink->imk", kj, dkernel_T) / n  # (d, M, K)

            # Conditional derivative: divide by p(z_j = ξ_k)
            dp_cond = dp_joint / (pj.view(1, M, 1) + 1e-12)  # (d, M, K)

            # u_ij = |∂p(z_i|z_j)/∂z_i|
            u = dp_cond.abs()  # (d, M, K)

            # Normalise to density p̃_ij
            norms = u.sum(dim=2, keepdim=True) * dz  # (d, M, 1)
            p_tilde = u / (norms + 1e-12)            # (d, M, K)

            # Generalized JSD: H(mean) − mean(H)
            m_tilde = p_tilde.mean(dim=1)  # (d, K)
            H_m = -(m_tilde * torch.log(m_tilde + 1e-12)).sum(dim=1) * dz     # (d,)
            H_each = -(p_tilde * torch.log(p_tilde + 1e-12)).sum(dim=2) * dz  # (d, M)
            H_mean = H_each.mean(dim=1)                                        # (d,)

            jsd_j = H_m - H_mean  # (d,)

            # Zero out the i==j diagonal entry: multiply by (1 - one_hot[j])
            # No in-place ops — eye is constant, subtraction creates a new tensor.
            total_jsd = total_jsd + (jsd_j * (1.0 - eye[j])).sum()

        return total_jsd

    def anticollapse_loss(self, z: torch.Tensor, sigma: float) -> torch.Tensor:
        """l_KL-uni = Σ_i KL(U(−√3, √3) ∥ p(z_i)) — prevents collapse to Diracs."""
        n, d = z.shape
        sqrt3 = 3.0 ** 0.5
        K = self.K

        # Deterministic grid over U(−√3, √3) — matches paper Eq. 16 where the
        # expectation E_U[log p(z_i)] is estimated as an average over K samples.
        # Using a fixed grid (instead of torch.rand) removes stochastic noise so
        # this term is directly comparable to the deterministic uni/biv terms.
        u_samples = torch.linspace(-sqrt3, sqrt3, K, device=z.device)  # (K,)

        # KDE at uniform samples for ALL dims at once
        # diff: (d, K, n) = u_samples[k] − z[l, i]
        diff = u_samples.view(1, K, 1) - z.T.unsqueeze(1)  # (d, K, n)
        kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (
            sigma * (2 * torch.pi) ** 0.5
        )  # (d, K, n)
        p_z = kernel.mean(dim=2)  # (d, K)

        # E_U[log p(z_i)] per dimension
        log_p = torch.log(p_z + 1e-12)   # (d, K)
        E_log_p = log_p.mean(dim=1)      # (d,)

        log_range = torch.log(torch.tensor(2 * sqrt3, device=z.device))
        kl = -log_range - E_log_p        # (d,)
        return kl.sum()

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L_Cliff = λ_uni · l_uni + λ_biv · l_biv + λ_KL-uni · l_KL-uni.

        Args:
            z: (n, d_raw) tensor of pooled encoder features (will be projected
               to ``latent_dim`` before computing losses).
        """
        # Force float32 — KDE involves exp/log that are numerically fragile
        # in float16 under AMP (exp underflows → 0, log(0) → -inf → NaN).
        with torch.amp.autocast("cuda", enabled=False):
            z = z.float()

            # Learnable projection: (n, d_raw) → (n, latent_dim)
            self._ensure_proj(z.shape[1], z.device)
            z = self.proj(z)

            z_std = self._standardize(z)
            n = z_std.shape[0]

            # Adaptive bandwidth via Silverman's rule (or user override)
            sigma = self._bandwidth(n)

            # Shared grid for univariate / bivariate integration
            dz = (self.z_max - self.z_min) / self.K
            grid = torch.linspace(
                self.z_min + dz / 2, self.z_max - dz / 2, self.K, device=z.device
            )

            # Precompute KDE kernels once (reused by uni + biv)
            kernel, dkernel = self._kde_kernels(z_std, grid, sigma)

            l_uni = self.univariate_cliff_loss(z_std, grid, dz, dkernel)
            l_biv = self.bivariate_cliff_loss(z_std, grid, dz, kernel, dkernel, sigma)
            l_kl = self.anticollapse_loss(z_std, sigma)

            loss = (
                self.lambda_uni * l_uni
                + self.lambda_biv * l_biv
                + self.lambda_kl_uni * l_kl
            )

        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Univariate"] = l_uni
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Bivariate"] = l_biv
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-AntiCollapse"] = l_kl
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Total"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries