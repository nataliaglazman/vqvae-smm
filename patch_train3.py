import re

with open('train.py', 'r') as f:
    code = f.read()

# 1. Update validation loop to use masks
val_search = r'''            recon, diffs, \*_ = _model_eval\(images\)\s*
            # Use only cheap pixel \(L1\) loss for validation — skip perceptual,\s*
            # FFT and GDL which are expensive and unnecessary for tracking val\s*
            # performance\.\s*
            loss = torch\.nn\.functional\.l1_loss\(recon, images\)'''

val_replace = '''            recon, diffs, *_ = _model_eval(images)
            # Use only cheap pixel (L1) loss for validation, masked if available.
            if "mask" in batch:
                mask = batch["mask"].to(device, non_blocking=True)
                loss = torch.nn.functional.l1_loss(recon * mask, images * mask, reduction="sum") / mask.sum().clamp(min=1e-5)
            else:
                loss = torch.nn.functional.l1_loss(recon, images)'''

code = re.sub(val_search, val_replace, code)

# 2. Update training loop to pass mask to BaselineLoss
train_search = r'''                    net_out = \{"reconstruction": \[recon\], "quantization_losses": \[\]\}\s*
                    recon_loss = loss_fn\(net_out, images\) \* args\.scale_recon_loss'''

train_replace = '''                    net_out = {"reconstruction": [recon], "quantization_losses": []}
                    if "mask" in batch:
                        net_out["mask"] = batch["mask"].to(device, non_blocking=True)
                    recon_loss = loss_fn(net_out, images) * args.scale_recon_loss'''

code = re.sub(train_search, train_replace, code)

with open('train.py', 'w') as f:
    f.write(code)

