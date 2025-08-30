#8x upscaling with Gaussian smoothing post-processing

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
from skimage.transform import resize
from scipy.ndimage import generic_filter, gaussian_filter
import rasterio
from rasterio.transform import Affine

# Dataset Class
class ElevationSuperResDataset(Dataset):
    def __init__(self, tif_path, patch_size=256, scale=8, max_nan_pct=0.05):
        with rasterio.open(tif_path) as src:
            self.hr_raw = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs

        self.mask = np.isnan(self.hr_raw)
        self.hr_filled = self._fill_nan_local_median(self.hr_raw, ksize=3)
        smoothed = gaussian_filter(self.hr_filled, sigma=0.5)

        self.min_val = np.nanmin(smoothed)
        self.max_val = np.nanmax(smoothed)
        self.hr = (smoothed - self.min_val) / (self.max_val - self.min_val)
        self.lr = resize(self.hr, (self.hr.shape[0] // scale, self.hr.shape[1] // scale), anti_aliasing=True)

        self.scale = scale
        self.patch_size = patch_size
        self.max_nan_pct = max_nan_pct

    def _fill_nan_local_median(self, array, ksize=5):
        def nanmedian_filter(values):
            v = values[~np.isnan(values)]
            return np.median(v) if v.size > 0 else 0.0
        return generic_filter(array, nanmedian_filter, size=ksize, mode='reflect')

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        if self.lr.shape[0] < self.patch_size or self.lr.shape[1] < self.patch_size:
            lr_patch = self.lr
            hr_patch = self.hr
        else:
            for _ in range(10):
                x = np.random.randint(0, self.lr.shape[0] - self.patch_size)
                y = np.random.randint(0, self.lr.shape[1] - self.patch_size)
                x_hr, y_hr = x * self.scale, y * self.scale
                hr_patch = self.hr[x_hr:x_hr + self.patch_size * self.scale,
                                   y_hr:y_hr + self.patch_size * self.scale]
                if np.isnan(hr_patch).mean() < self.max_nan_pct:
                    break
            lr_patch = self.lr[x:x + self.patch_size, y:y + self.patch_size]

        return torch.from_numpy(lr_patch).float().unsqueeze(0), torch.from_numpy(hr_patch).float().unsqueeze(0)

# Elevation-Aware Loss
class ElevationGradientLoss(nn.Module):
    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()
        l1_loss = (pred - target).abs().mean()
        return l1_loss + 0.5 * grad_loss

@torch.no_grad()
def infer_fully_convolutional(model, dataset, output_path='fullEL8X.tif'):
    device = next(model.parameters()).device
    model.eval()

    lr = dataset.lr
    scale = dataset.scale
    h_lr, w_lr = lr.shape
    h_hr, w_hr = h_lr * scale, w_lr * scale

    input_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(device)
    sr_tensor = model(input_tensor).squeeze().cpu().numpy()

    # Gaussian post-filtering
    sr_tensor = gaussian_filter(sr_tensor, sigma=0.8)

    sr_denorm = sr_tensor * (dataset.max_val - dataset.min_val) + dataset.min_val
    sr_denorm = np.clip(sr_denorm, dataset.min_val, dataset.max_val)

    original_transform = dataset.transform
    new_transform = Affine(
        original_transform.a / scale, original_transform.b, original_transform.c,
        original_transform.d, original_transform.e / scale, original_transform.f
    )

    with rasterio.open(
        output_path, 'w', driver='GTiff', height=sr_denorm.shape[0], width=sr_denorm.shape[1],
        count=1, dtype=sr_denorm.dtype, crs=dataset.crs, transform=new_transform
    ) as dst:
        dst.write(sr_denorm, 1)

    print(f"✅ Fully convolutional super-res output saved to: {output_path}")

# Main Training + Inference
def train_and_infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ElevationSuperResDataset("elevation.tif", patch_size=256, scale=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SwinIR(
        upscale=8,
        in_chans=1,
        img_size=256,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    criterion = ElevationGradientLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):
        model.train()
        total_loss = 0
        for lr, hr in tqdm(dataloader, desc=f"Epoch {epoch+1}/100"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            min_h = min(sr.shape[2], hr.shape[2])
            min_w = min(sr.shape[3], hr.shape[3])
            sr = sr[:, :, :min_h, :min_w]
            hr = hr[:, :, :min_h, :min_w]
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    infer_fully_convolutional(model, dataset)

if __name__ == '__main__':
    train_and_infer()
