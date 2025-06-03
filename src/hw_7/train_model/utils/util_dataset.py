import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.fh_utils import *

class FreiHand2DDataset(Dataset):
    def __init__(self, base_path, split='training', version=sample_version.gs, transform=None):
        super().__init__()
        self.base_path = base_path
        self.split = split
        self.version = version
        self.transform = transform

        # Load all annotations
        self.db_data_anno = load_db_annotation(base_path, split)
        self.num_samples = db_size(split)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read image and mask
        img = read_img(idx, self.base_path, self.split, self.version)  # shape: H x W x 3
        msk = read_msk(idx, self.base_path)                            # shape: H x W

        # Get annotations
        K, mano, xyz = self.db_data_anno[idx]
        K, xyz = np.array(K), np.array(xyz)

        # Project 3D joints to 2D
        uv = projectPoints(xyz, K).astype(np.float32)  # shape: (21, 2)

        # Optionally apply transform
        if self.transform is not None:
            img, msk, uv = self.transform(img, msk, uv)

        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
        msk = msk[:, :, 0]
        msk = (msk > 127).astype(np.float32) 
        msk = torch.from_numpy(msk).unsqueeze(0).float()             # 1 x H x W
        # uv = torch.from_numpy(uv).float()                            # 21 x 2

        return {
            'image': img,
            'mask': msk,
            'joints_uv': uv,
            'index': idx
        }

# --- DataLoader example usage ---
def get_dataloader(dataset, batch_size=16, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader

def generate_heatmaps(joints_uv, image_size=224, sigma=2):
    B, num_joints, _ = joints_uv.shape
    heatmaps = torch.zeros((B, num_joints, image_size, image_size), dtype=torch.float32)

    for b in range(B):
        for j in range(num_joints):
            x, y = joints_uv[b, j]
            if x < 0 or y < 0 or x >= image_size or y >= image_size:
                continue  # skip out-of-bounds

            # Create Gaussian
            xx, yy = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            xx = xx.to(dtype=torch.float32)
            yy = yy.to(dtype=torch.float32)

            heatmap = torch.exp(-((xx - y)**2 + (yy - x)**2) / (2 * sigma**2))
            heatmaps[b, j] = heatmap

    return heatmaps

def heatmaps_to_coords(heatmaps):
    B, C, H, W = heatmaps.shape
    heatmaps = heatmaps.reshape(B, C, -1)
    coords = heatmaps.argmax(dim=-1)  # [B, C]
    coords_y = coords // W
    coords_x = coords % W
    return torch.stack([coords_x, coords_y], dim=-1).float()  # [B, 21, 2]

