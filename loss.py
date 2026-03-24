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
    def __init__(self, commitment_weight: float = 0.25):
        super(BaselineLoss, self).__init__()

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

class CliffLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_uni: float = 1.0,
        lambda_biv: float = 1.0,
        lambda_kl_uni: float = 1.0,
        sigma: float = 0.1,
        K: int = 100,
        M: int = 10,
        z_min: float = -5.0,
        z_max: float = 5.0,
    ):
        super(CliffLoss, self).__init__()
        self.lambda_uni = lambda_uni
        self.lambda_biv = lambda_biv
        self.lambda_kl_uni = lambda_kl_uni
        self.sigma = sigma
        self.K = K
        self.M = M
        self.z_min = z_min
        self.z_max = z_max
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}
    def univariate_cliff_loss(self, z, sigma=0.1, K=100, z_min=-5.0, z_max=5.0):
        """
        Computes l_uni = sum_i H(s_i), the entropy of the normalized
        gradient magnitude of each marginal density.
        
        Args:
            z: (n, d) tensor of latent factors (already standardized to mean=0, std=1)
            sigma: KDE bandwidth
            K: number of grid points for numerical integration
            z_min, z_max: integration bounds
        """
        n, d = z.shape
        
        # Grid points for numerical integration: (K,)
        dz = (z_max - z_min) / K
        grid = torch.linspace(z_min + dz/2, z_max - dz/2, K, device=z.device)
        
        total_entropy = 0.0
        
        for i in range(d):
            zi = z[:, i]  # (n,)
            
            # KDE marginal density at grid points: p(zi)
            # diff: (K, n) — grid points minus each sample
            diff = grid.unsqueeze(1) - zi.unsqueeze(0)  # (K, n)
            
            # Gaussian kernel values: (K, n)
            kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
            
            # Derivative of each kernel w.r.t. zi: d/dz N(z; mu, sigma^2) = -(z-mu)/sigma^2 * N(...)
            dkernel = (-diff / sigma**2) * kernel  # (K, n)
            
            # dp/dz_i at each grid point: (K,)
            dp_dzi = dkernel.mean(dim=1)
            
            # Absolute gradient magnitude: |dp/dz_i|
            abs_grad = dp_dzi.abs()  # (K,)
            
            # Normalization constant c = integral of |dp/dz_i| over z_i
            c = abs_grad.sum() * dz
            
            # Normalized density s_i(z_i) = |dp/dz_i| / c
            si = abs_grad / (c + 1e-12)
            
            # Differential entropy H(s_i) = -integral s_i * log(s_i) dz_i
            log_si = torch.log(si + 1e-12)
            H_si = -(si * log_si).sum() * dz
            
            total_entropy += H_si
        
        return total_entropy  # minimize this
    
    def bivariate_cliff_loss(self, z, sigma=0.1, K=100, M=10, z_min=-5.0, z_max=5.0):
        """
        Computes l_biv = sum_{i} sum_{j!=i} JSD[p̃_ij(·|z_j=ξ_1), ..., p̃_ij(·|z_j=ξ_M)]
        
        Args:
            z: (n, d) standardized latent factors
            sigma: KDE bandwidth
            K: grid points for numerical integration
            M: number of conditioning values ξ_k sampled from the batch
        """
        n, d = z.shape
        dz = (z_max - z_min) / K
        grid = z_min + dz * (torch.arange(K, device=z.device) + 0.5)

        total_jsd = 0.0

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue

                zi = z[:, i]  # (n,)
                zj = z[:, j]  # (n,)

                # Sample M conditioning values ξ_k from the batch
                indices = torch.randperm(n, device=z.device)[:M]
                xi_vals = zj[indices]  # (M,)

                # --- z_i dimension: kernel and its derivative on the grid ---
                diff_i = grid.unsqueeze(1) - zi.unsqueeze(0)       # (K, n)
                kernel_i = torch.exp(-0.5 * (diff_i / sigma) ** 2) \
                            / (sigma * (2 * torch.pi) ** 0.5)       # (K, n)
                dkernel_i = (-diff_i / sigma ** 2) * kernel_i       # (K, n)

                # --- z_j dimension: kernel at each ξ_k for every sample ---
                diff_j = xi_vals.unsqueeze(1) - zj.unsqueeze(0)    # (M, n)
                kernel_j = torch.exp(-0.5 * (diff_j / sigma) ** 2) \
                            / (sigma * (2 * torch.pi) ** 0.5)       # (M, n)

                # Marginal p(z_j = ξ_k) via 1D KDE (Eq. 2)
                p_zj = kernel_j.mean(dim=1)                         # (M,)

                # Joint partial derivative ∂p(z_i, z_j)/∂z_i at (grid, ξ_k) (Eq. 5)
                # For each (m, k): (1/n) Σ_l dkernel_i[k, l] * kernel_j[m, l]
                dp_joint_dzi = kernel_j @ dkernel_i.T / n           # (M, K)

                # Conditional derivative (Eq. 11):
                # ∂p(z_i|z_j)/∂z_i = (1/p(z_j)) * ∂p(z_i,z_j)/∂z_i
                dp_cond_dzi = dp_joint_dzi / (p_zj.unsqueeze(1) + 1e-12)  # (M, K)

                # u_ij = |∂p(z_i|z_j)/∂z_i| (Eq. 12)
                u_ij = dp_cond_dzi.abs()                            # (M, K)

                # Normalize to density p̃_ij (Eq. 13)
                norms = u_ij.sum(dim=1, keepdim=True) * dz          # (M, 1)
                p_tilde = u_ij / (norms + 1e-12)                    # (M, K)

                # --- Generalized JSD (Eq. 14) ---
                # m̃(z_i) = (1/M) Σ_k p̃_ij(z_i | z_j = ξ_k)
                m_tilde = p_tilde.mean(dim=0)                       # (K,)

                # H(m̃)
                H_m = -(m_tilde * torch.log(m_tilde + 1e-12)).sum() * dz

                # (1/M) Σ_k H(p̃_ij(·|z_j = ξ_k))
                H_each = -(p_tilde * torch.log(p_tilde + 1e-12)).sum(dim=1) * dz  # (M,)
                H_mean = H_each.mean()

                # JSD = H(m̃) - (1/M) Σ H(p̃_k)
                total_jsd += (H_m - H_mean)

        return total_jsd
    
    def anticollapse_loss(self, z, sigma=0.1, K=100, z_min=-5.0, z_max=5.0):
        """
        Computes l_KL-uni = sum_i KL(U(-sqrt(3), sqrt(3)) || p(z_i))
        
        Encourages each standardized marginal p(z_i) to be close to a
        uniform distribution, preventing collapse to Diracs.
        
        Args:
            z: (n, d) standardized latent factors
            sigma: KDE bandwidth
            K: number of sample points for estimating the expectation
        """
        n, d = z.shape
        sqrt3 = 3.0 ** 0.5

        total_kl = 0.0

        for i in range(d):
            zi = z[:, i]  # (n,)

            # Sample K points from U(-sqrt(3), sqrt(3))
            u_samples = torch.rand(K, device=z.device) * 2 * sqrt3 - sqrt3  # (K,)

            # Evaluate KDE marginal p(z_i) at those uniform samples (Eq. 2)
            diff = u_samples.unsqueeze(1) - zi.unsqueeze(0)  # (K, n)
            kernel = torch.exp(-0.5 * (diff / sigma) ** 2) \
                    / (sigma * (2 * torch.pi) ** 0.5)        # (K, n)
            p_zi = kernel.mean(dim=1)                          # (K,)


            log_p = torch.log(p_zi + 1e-12)  # (K,)
            E_log_p = log_p.mean()  # Monte Carlo estimate of E_U[log p(z_i)]

            kl = -torch.log(torch.tensor(2 * sqrt3, device=z.device)) - E_log_p
            total_kl += kl

        return total_kl  # minimize this
    
    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize each factor to zero mean and unit variance (per-batch)."""
        mu = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (z - mu) / std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L_Cliff = λ_uni * l_uni + λ_biv * l_biv + λ_KL-uni * l_KL-uni.

        Args:
            z: (n, d) tensor of latent factors from the encoder.
        """
        z_std = self._standardize(z)

        l_uni = self.univariate_cliff_loss(
            z_std, sigma=self.sigma, K=self.K, z_min=self.z_min, z_max=self.z_max
        )
        l_biv = self.bivariate_cliff_loss(
            z_std, sigma=self.sigma, K=self.K, M=self.M, z_min=self.z_min, z_max=self.z_max
        )
        l_kl = self.anticollapse_loss(
            z_std, sigma=self.sigma, K=self.K, z_min=self.z_min, z_max=self.z_max
        )

        loss = self.lambda_uni * l_uni + self.lambda_biv * l_biv + self.lambda_kl_uni * l_kl

        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Univariate"] = l_uni
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Bivariate"] = l_biv
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-AntiCollapse"] = l_kl
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Cliff-Total"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries