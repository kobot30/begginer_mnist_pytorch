from torch.utils.data import Dataset
import cv2
import numpy as np


def get_image_mask_set(mnist_images):
    images = []
    masks = []
    for image in mnist_images:
            image = cv2.resize(image, (56, 56)) # 56x56ピクセルにリサイズ
            images.append(image)
                
            _, binary = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
            mask = (binary > 0).astype(np.float32)
            no_mask = (binary == 0).astype(np.float32)
            mask_array = np.asarray([no_mask, mask]) # channel=2 (背景、検出領域)
            masks.append(mask_array)
    
    return images, masks


class MnistDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)

        return [image, mask]
