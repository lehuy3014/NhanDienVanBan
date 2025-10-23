import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load dictionary
dict_path = os.path.join("vietnamese", "vn_dictionary.txt")  
with open(dict_path, encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
vocab = sorted(set("".join(lines))) + ['<PAD>', '<SOS>', '<EOS>']
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}

MAX_LEN = 36

def polygon_to_box(polygon):
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def encode_text(text):
    tokens = ['<SOS>'] + list(text) + ['<EOS>']
    indices = [char2idx.get(c, char2idx['<PAD>']) for c in tokens]
    indices += [char2idx['<PAD>']] * (MAX_LEN - len(indices))
    return torch.tensor(indices[:MAX_LEN])

class OCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        for label_file in sorted(glob.glob(f"{label_dir}/gt_*.txt")):
            img_id = os.path.splitext(os.path.basename(label_file))[0].split('_')[-1]
            img_path = os.path.join(image_dir, f"im{int(img_id):04d}.jpg")
            if not os.path.exists(img_path):
                continue
            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    polygon = list(map(int, parts[:8]))
                    text = parts[8]
                    if text == "###" or not text.strip(): continue
                    box = polygon_to_box(polygon)
                    self.samples.append((img_path, box, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (x1, y1, x2, y2), text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        cropped = image.crop((x1, y1, x2, y2))
        if self.transform:
            cropped = self.transform(cropped)
        else:
            cropped = transforms.ToTensor()(cropped)
        target = encode_text(text)
        return cropped, target
