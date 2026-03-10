
from config import ADNIDataset
from vqvae import VQVAE2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from loss import BaurLoss, BaselineLoss

def train(args):
    # Build MONAI transforms for training and validation

    device = args.device

    data = ADNIDataset(args.dataroot, args.val_size, args.test_size, args.spacing, args.spatial_size)
    data = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    num_epochs = args.epochs
    batch_size = args.batch_size

    model = VQVAE2()
    params = model.parameters()
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=use_fused)
    recon_loss_fn = BaselineLoss().to(device)

    for i in range(num_epochs):
        for batch in data:
            images = batch["image_t1"]  # Assuming image_t1 is the input
            recon_images, diffs, encoder_features, decoder_outputs, id_outputs = model(images)

            # Total loss
            loss = recon_loss_fn(recon_images, images)
            vq_loss = sum(diffs) * args.vq_loss_weight
            loss += vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()