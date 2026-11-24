import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 512


def load_image(path):
    return Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


class AffectNetDataset(Dataset):
    """AffectNet images listed in the annotation CSV (basic expressions only)."""

    def __init__(self, csv_path, image_dir, neutral_only=False):
        self.image_dir = image_dir
        meta = pd.read_csv(csv_path, delimiter=',')
        meta = meta[meta['expression'] == 0] if neutral_only else meta[meta['expression'] <= 6]

        # Keep only rows whose image exists on disk
        self.files = [
            f.split('/')[-1] for f in meta['subDirectory_filePath']
            if os.path.exists(os.path.join(image_dir, f.split('/')[-1]))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        return load_image(os.path.join(self.image_dir, fname)), fname


class ImageFolderDataset(Dataset):
    """Unlabeled flat-directory dataset (e.g. FFHQ, CelebA-HQ)."""

    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.EXTS)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        return load_image(os.path.join(self.img_dir, fname)), fname


def make_dataset(img_dir, affectnet_csv=None, neutral_only=False):
    """AffectNet if an annotation CSV is given (images in img_dir), else a plain image folder."""
    if affectnet_csv:
        return AffectNetDataset(affectnet_csv, img_dir, neutral_only=neutral_only)
    return ImageFolderDataset(img_dir)
