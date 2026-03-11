import copy
from math import log2
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from helper import HelperModule


class ReZero(HelperModule):
    """3D ReZero residual block with learnable scaling parameter."""

    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(HelperModule):
    """Stack of 3D ReZero residual blocks with optional gradient checkpointing."""

    def build(
        self,
        in_channels: int,
        res_channels: int,
        nb_layers: int,
        use_checkpoint: bool = True,
    ):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) for _ in range(nb_layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_checkpoint and self.training:
            x = checkpoint(self.stack, x, use_reentrant=False)
        else:
            x = self.stack(x)
        return x


class Encoder(HelperModule):
    """3D Encoder with strided convolutions for downsampling."""

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        nb_res_layers: int,
        downscale_factor: int,
        use_checkpoint: bool = True,
    ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.BatchNorm3d(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv3d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers, use_checkpoint=use_checkpoint))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(HelperModule):
    """3D Decoder with transposed convolutions for upsampling.
    """

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        res_channels: int,
        nb_res_layers: int,
        upscale_factor: int,
        use_checkpoint: bool = True,
    ):
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))

        layers = [nn.Conv3d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(
            ResidualStack(
                hidden_channels,
                res_channels,
                nb_res_layers,
                use_checkpoint=use_checkpoint,
            )
        )
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.BatchNorm3d(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, out_channels


        layers.append(nn.Conv3d(c_channel, out_channels, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        feat = self.layers(x)
        return feat
    


class CodeLayer(HelperModule):
    """3D Vector Quantization layer with EMA codebook updates.

    Includes codebook utilization monitoring (perplexity) and automatic
    dead-code resetting.  Dead entries — those whose EMA cluster size falls
    below ``dead_threshold`` — are re-initialised to randomly sampled encoder
    outputs so they can be re-used.
    """

    def build(self, in_channels: int, embed_dim: int, nb_entries: int,
              dead_threshold: float = 1.0):
        self.conv_in = nn.Conv3d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5
        self.dead_threshold = dead_threshold

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        # x: (B, C, D, H, W) -> (B, D, H, W, embed_dim)
        x = self.conv_in(x.float()).permute(0, 2, 3, 4, 1)
        flatten = x.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # ── Dead code reset ──────────────────────────────────────────
            # Entries whose EMA cluster size is below the threshold are
            # considered dead.  Replace them with randomly sampled encoder
            # outputs so they have a chance of being picked up again.
            dead_mask = self.cluster_size < self.dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                n_samples = flatten.shape[0]
                replace_idx = torch.randint(0, n_samples, (n_dead,), device=flatten.device)
                self.embed.data[:, dead_mask] = flatten[replace_idx].T
                self.embed_avg.data[:, dead_mask] = flatten[replace_idx].T
                self.cluster_size.data[dead_mask] = 1.0

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 4, 1, 2, 3), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    @torch.no_grad()
    def codebook_utilization(self) -> dict:
        """Compute codebook health metrics (call after forward pass).

        Returns:
            dict with keys:
              - ``active_codes``: number of entries with cluster_size >= dead_threshold
              - ``utilization``:  fraction of codebook in use (0–1)
              - ``perplexity``:   exp(entropy) of the usage distribution — equals
                                  n_embed when all codes are used uniformly
        """
        # Normalise cluster sizes into a probability distribution
        probs = self.cluster_size / self.cluster_size.sum().clamp(min=1e-10)
        # Perplexity = exp(H) where H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        perplexity = torch.exp(-(probs * log_probs).sum())
        active = (self.cluster_size >= self.dead_threshold).sum()
        return {
            "active_codes": active.item(),
            "utilization": active.item() / self.n_embed,
            "perplexity": perplexity.item(),
        }

class Upscaler(HelperModule):
    """3D Upscaler for hierarchical code conditioning."""

    def build(
        self,
        embed_dim: int,
        scaling_rates: list[int],
    ):
        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose3d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm3d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)


class VQVAE(HelperModule):
    def build(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        res_channels: int = 32,
        nb_res_layers: int = 2,
        nb_levels: int = 3,
        embed_dim: int = 32,
        nb_entries: int = 384,
        scaling_rates: list[int] = [2, 2, 2],
        use_checkpoint: bool = True,
    ):
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"
        self.nb_levels = nb_levels
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    in_channels,
                    hidden_channels,
                    res_channels,
                    nb_res_layers,
                    scaling_rates[0],
                    use_checkpoint,
                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(
                Encoder(
                    hidden_channels,
                    hidden_channels,
                    res_channels,
                    nb_res_layers,
                    sr,
                    use_checkpoint,
                )
            )


        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels + embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    embed_dim * nb_levels,
                    hidden_channels,
                    in_channels,
                    res_channels,
                    nb_res_layers,
                    scaling_rates[0],
                    use_checkpoint,

                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(
                Decoder(
                    embed_dim * (nb_levels - 1 - i),
                    hidden_channels,
                    embed_dim,
                    res_channels,
                    nb_res_layers,
                    sr,
                    use_checkpoint,
                )
            )

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            rates = scaling_rates[1 : len(scaling_rates) - i][::-1]  # noqa: E203
            self.upscalers.append(Upscaler(embed_dim, rates))

    def forward(self, x, return_recon=True, pool_only=False, n_views=1, subsets=None):
        """Forward pass through VQ-VAE-2.

        Args:
            x: Input tensor (B, C, D, H, W)
            return_recon: If False, skip decoder for memory efficiency (contrastive-only mode)
            pool_only: If True, return per-level pooled (B, C) vectors instead of spatial maps.
            n_views: Number of views (required when content_proj is active).
            subsets: View subsets for Gumbel mask (required when content_proj is active).

        Returns:
            final_output: Reconstruction (or None if return_recon=False)
            diffs: VQ commitment losses per level
            encoder_features: Per-level encoder features.
                              If pool_only=True:  list of (B, C) pooled vectors
                                  Level 0 has content_channels dims (masked),
                                  levels 1+ have hidden_channels dims.
                              If pool_only=False: list of (B, C, D, H, W) spatial maps
            estimated_content_indices: Content channel indices from the Gumbel mask
                                       (None if content_proj not configured)
            decoder_outputs: Decoder features per level (or empty list)
            id_outputs: Codebook indices per level
        """
        assert x.ndim == 5, f"Expected 5D input (B, C, D, H, W), got {x.ndim}D with shape {x.shape}"
        assert x.shape[1] == self.encoders[0].layers[0][0].in_channels, (
            f"Expected {self.encoders[0].layers[0][0].in_channels} input channels, got {x.shape[1]}"
        )
        input_shape = x.shape[2:]  # (D, H, W) – remember for final interpolation
        encoder_outputs = []  # Spatial (5D) feature maps, consumed by codebook/decoder loop
        encoder_pools = []  # Pooled (B, C) vectors, returned for contrastive loss
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []
        # Encoder forward pass
        enc_input = x
        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        del x, enc_input

        for l in range(self.nb_levels - 1, -1, -1):
            codebook = self.codebooks[l]

            enc_out = encoder_outputs[l]
            encoder_outputs[l] = None  # release spatial map reference as soon as consumed
            expected_in = codebook.conv_in.in_channels

            if len(decoder_outputs) and return_recon:
                # Interpolate decoder output to match encoder output size if needed
                dec_out = decoder_outputs[-1]
                if dec_out.shape[2:] != enc_out.shape[2:]:
                    dec_out = F.interpolate(
                        dec_out,
                        size=enc_out.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                combined = torch.cat([enc_out, dec_out], dim=1)
                del enc_out
                code_q, code_d, emb_id = codebook(combined)
                del combined
            else:
                # If this codebook expects conditioning channels, pad with zeros when reconstruction is skipped
                if expected_in > enc_out.shape[1]:
                    cond_channels = expected_in - enc_out.shape[1]
                    zeros = torch.zeros(
                        enc_out.shape[0],
                        cond_channels,
                        *enc_out.shape[2:],
                        device=enc_out.device,
                        dtype=enc_out.dtype,
                    )
                    enc_out = torch.cat([enc_out, zeros], dim=1)
                code_q, code_d, emb_id = codebook(enc_out)
                del enc_out

            diffs.append(code_d)
            id_outputs.append(emb_id)

            if return_recon:
                decoder = self.decoders[l]
                # Upscale previous code outputs and interpolate to match current level size
                upscaled_codes = []
                target_size = code_q.shape[2:]
                for i, c in enumerate(code_outputs):
                    upscaled = self.upscalers[i](c, upscale_counts[i])
                    if upscaled.shape[2:] != target_size:
                        upscaled = F.interpolate(
                            upscaled,
                            size=target_size,
                            mode="trilinear",
                            align_corners=False,
                        )
                    upscaled_codes.append(upscaled)
                code_outputs = upscaled_codes
                upscale_counts = [u + 1 for u in upscale_counts]

                decoder_in = torch.cat([code_q, *code_outputs], axis=1)

                decoder_outputs.append(decoder(decoder_in))

                code_outputs.append(code_q)
                upscale_counts.append(0)

        if return_recon:
            final_output = decoder_outputs[-1]
            # Strided conv/transposed-conv pairs lose fractional pixels on odd
            # spatial dims.  Interpolate back to the original input shape.
            if final_output.shape[2:] != input_shape:
                final_output = F.interpolate(
                    final_output,
                    size=input_shape,
                    mode="trilinear",
                    align_corners=False,
                )
        else:
            final_output = None
            decoder_outputs = []

        # Return pooled features (memory-efficient) or full spatial maps
        encoder_features = encoder_pools if pool_only else encoder_outputs

        return (
            final_output,
            diffs,
            encoder_features,
            decoder_outputs,
            id_outputs,
        )

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]
