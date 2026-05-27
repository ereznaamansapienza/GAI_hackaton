import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

MANIFEST_PATH = "manifest.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-4

# ============================================================
# DATASET
# ============================================================

MAX_TRACKS = 4

class MusicDataset(Dataset):
    def __init__(self, manifest, split="train"):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)

    def load_roll(self, path):

        data = np.load(path)

        pitched = data["pitched"].astype(np.float32)
        drum = data["drum"].astype(np.float32)

        return pitched, drum

    def pad_tracks(self, pitched):

        n_tracks = pitched.shape[0]

        if n_tracks > MAX_TRACKS:
            pitched = pitched[:MAX_TRACKS]

        elif n_tracks < MAX_TRACKS:

            pad = np.zeros(
                (
                    MAX_TRACKS - n_tracks,
                    128,
                    128,
                ),
                dtype=np.float32,
            )

            pitched = np.concatenate(
                [pitched, pad],
                axis=0,
            )

        return pitched

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        X_pitch, X_drum = self.load_roll(row["X_path"])
        Z_pitch, Z_drum = self.load_roll(row["Z_path"])
        Y_pitch, Y_drum = self.load_roll(row["Y_path"])

        # PAD TRACKS
        X_pitch = self.pad_tracks(X_pitch)
        Z_pitch = self.pad_tracks(Z_pitch)
        Y_pitch = self.pad_tracks(Y_pitch)

        # add drum as extra channel
        X = np.concatenate([X_pitch, X_drum[None]], axis=0)
        Z = np.concatenate([Z_pitch, Z_drum[None]], axis=0)
        Y = np.concatenate([Y_pitch, Y_drum[None]], axis=0)

        return (
            torch.tensor(X),
            torch.tensor(Z),
            torch.tensor(Y),
        )

# ============================================================
# MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class StyleTransferNet(nn.Module):
    def __init__(self, in_channels=10):
        super().__init__()

        # content encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)

        # style encoder
        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.style_fc = nn.Linear(64, 256)

        # decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, z):

        # content
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        latent = self.enc3(p2)

        # style embedding
        s = self.style_encoder(z).flatten(1)
        s = self.style_fc(s)

        # FiLM-style conditioning
        s = s[:, :, None, None]
        latent = latent + s

        # decoder
        u1 = self.up1(latent)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2)

        out = self.final(u2)

        return torch.sigmoid(out)

# ============================================================
# CHROMA LOSS
# ============================================================

def chroma(x):
    """
    x: (B,C,128,128)
    """
    pitch_energy = x.sum(dim=1)

    chroma = torch.zeros(
        x.shape[0],
        12,
        x.shape[-1],
        device=x.device
    )

    for p in range(128):
        chroma[:, p % 12] += pitch_energy[:, p]

    return chroma

def chroma_loss(pred, target):
    c1 = chroma(pred)
    c2 = chroma(target)

    c1 = F.normalize(c1.flatten(1), dim=1)
    c2 = F.normalize(c2.flatten(1), dim=1)

    sim = (c1 * c2).sum(dim=1)

    return 1 - sim.mean()

# ============================================================
# STYLE LOSS
# ============================================================

def histogram_loss(pred, target):

    pred_hist = pred.mean(dim=(-1, -2))
    targ_hist = target.mean(dim=(-1, -2))

    pred_hist = F.normalize(pred_hist, dim=1)
    targ_hist = F.normalize(targ_hist, dim=1)

    sim = (pred_hist * targ_hist).sum(dim=1)

    return 1 - sim.mean()

# ============================================================
# TRAINING
# ============================================================

manifest = pd.read_csv(MANIFEST_PATH)

train_ds = MusicDataset(manifest, "train")

val_ds = MusicDataset(manifest, "val")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
)

sample_x, _, _ = train_ds[0]
CHANNELS = sample_x.shape[0]

model = StyleTransferNet(CHANNELS).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
)

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X, Z, Y in tqdm(train_loader):

        X = X.to(DEVICE)
        Z = Z.to(DEVICE)
        Y = Y.to(DEVICE)

        pred = model(X, Z)

        recon = F.binary_cross_entropy(pred, Y)

        cp_loss = chroma_loss(pred, X)

        sf_loss = histogram_loss(pred, Z)

        loss = (
            recon
            + 0.5 * cp_loss
            + 0.5 * sf_loss
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X, Z, Y in val_loader:

            X = X.to(DEVICE)
            Z = Z.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X, Z)

            recon = F.binary_cross_entropy(pred, Y)

            cp_loss = chroma_loss(pred, X)

            sf_loss = histogram_loss(pred, Z)

            loss = (
                recon
                + 0.5 * cp_loss
                + 0.5 * sf_loss
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1} | "
        f"train={train_loss:.4f} "
        f"val={val_loss:.4f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), "music_style_transfer.pt")

print("Training complete.")

# Epoch 1 | train=0.6348 val=0.3232
# Training complete.