import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, resize_shape=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize_shape = resize_shape
        self.img_transform = transforms.Compose([
            transforms.Resize((resize_shape, resize_shape), interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        
        image_tensor = self.img_transform(image)
        mask = mask.resize((self.resize_shape, self.resize_shape), resample=Image.NEAREST)
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return image_tensor, mask_tensor
    

def voc_dataloaders(voc_root=r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012', batch_size=4):
    img_dir = os.path.join(voc_root, 'JPEGImages')
    mask_dir = os.path.join(voc_root, 'SegmentationClass')
    splits_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
    
    with open(os.path.join(splits_dir, 'train.txt'), 'r') as f:
        train_names = [line.strip() for line in f.readlines()]
        
    with open(os.path.join(splits_dir, 'val.txt'), 'r') as f:
        val_names = [line.strip() for line in f.readlines()]
        
    train_imgs = [os.path.join(img_dir, f"{name}.jpg") for name in train_names]
    train_masks = [os.path.join(mask_dir, f"{name}.png") for name in train_names]
    
    val_imgs = [os.path.join(img_dir, f"{name}.jpg") for name in val_names]
    val_masks = [os.path.join(mask_dir, f"{name}.png") for name in val_names]
    
    print(f" - 训练集: {len(train_imgs)} 张")
    print(f" - 验证集: {len(val_imgs)} 张")
    
    train_dataset = VOCDataset(train_imgs, train_masks)
    val_dataset = VOCDataset(val_imgs, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = voc_dataloaders()
    
    for images, masks in train_loader:
        print(f"Images shape: {images.shape} (预期: [Batch, 3, 512, 512])")
        print(f"Masks shape: {masks.shape}  (预期: [Batch, 512, 512], 值域应包含 0-20 或 255)")
        break