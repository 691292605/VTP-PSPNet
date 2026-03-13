import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import VTP_PSPNet 

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

def decode_segmap(image_idx, colormap=VOC_COLORMAP):
    r = np.zeros_like(image_idx).astype(np.uint8)
    g = np.zeros_like(image_idx).astype(np.uint8)
    b = np.zeros_like(image_idx).astype(np.uint8)
    for l in range(0, 21):
        idx = image_idx == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    idx_255 = image_idx == 255
    r[idx_255], g[idx_255], b[idx_255] = 224, 224, 192 
    return np.stack([r, g, b], axis=2)

def generate_cross_comparison(image_paths, large_weight_path, small_weight_path, save_format='png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 1. 加载 Large 模型 (512分辨率) =================
    print("正在加载 VTP-Large 模型...")
    # 传入 Large 的骨干权重路径 (这里假设与默认路径一致)
    model_large = VTP_PSPNet(
        num_classes=21, 
        config_path="./backbone_weight_large/config.json", 
        weight_path="./backbone_weight_large/model.safetensors"
    ).to(device) 
    ckpt_large = torch.load(large_weight_path, map_location=device)
    model_large.load_state_dict(ckpt_large['model_state_dict'] if 'model_state_dict' in ckpt_large else ckpt_large, strict=False)
    model_large.eval()

    # ================= 2. 加载 Small 模型 (1024分辨率) =================
    print("正在加载 VTP-Small 模型...")
    # 传入 Small 的骨干权重路径 (你需要确认这个文件夹存在)
    model_small = VTP_PSPNet(
        num_classes=21, 
        config_path="./backbone_weight/config.json", 
        weight_path="./backbone_weight/model.safetensors"
    ).to(device) 
    ckpt_small = torch.load(small_weight_path, map_location=device)
    model_small.load_state_dict(ckpt_small['model_state_dict'] if 'model_state_dict' in ckpt_small else ckpt_small, strict=False)
    model_small.eval()

    # ================= 3. 定义两套不同的预处理 =================
    transform_large = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_small = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_imgs = len(image_paths)
    ncols = 4  

    fig, axes = plt.subplots(nrows=num_imgs, ncols=ncols, figsize=(15, 4 * num_imgs), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
    if num_imgs == 1: axes = [axes]

    for i, image_path in enumerate(image_paths):
        print(f"正在处理第 {i+1}/{num_imgs} 张图片...")
        
        # 统一画图展示尺寸为 512x512
        display_size = (512, 512)
        
        original_img = Image.open(image_path).convert('RGB')
        
        # (a) 原图
        axes[i][0].imshow(np.array(original_img.resize(display_size))) 
        axes[i][0].axis('off')

        # (b) GT
        gt_path = image_path.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')
        if os.path.exists(gt_path):
            gt_mask = np.array(Image.open(gt_path).resize(display_size, Image.NEAREST))
            axes[i][1].imshow(decode_segmap(gt_mask))
        axes[i][1].axis('off')

        with torch.no_grad():
            # ================= 推理 Large 模型 (512) =================
            input_large = transform_large(original_img).unsqueeze(0).to(device)
            out_large = model_large(input_large) # shape: (1, 21, 512, 512)
            mask_large = torch.argmax(out_large.squeeze(0), dim=0).cpu().numpy()
            
            axes[i][2].imshow(decode_segmap(mask_large))
            axes[i][2].axis('off')

            # ================= 推理 Small 模型 (1024) =================
            input_small = transform_small(original_img).unsqueeze(0).to(device)
            out_small = model_small(input_small) # shape: (1, 21, 1024, 1024)
            mask_small_1024 = torch.argmax(out_small.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            
            # 【关键】将 1024 的掩码缩放回 512 以保证画图对比的一致性，使用最近邻插值保持类别不被破坏
            mask_small_512 = np.array(Image.fromarray(mask_small_1024).resize(display_size, Image.NEAREST))
            
            axes[i][3].imshow(decode_segmap(mask_small_512))
            axes[i][3].axis('off')

    labels = ["(a) Image", "(b) Ground Truth", "(c) VTP-Large (Baseline)", "(d) VTP-Small"]
    for col_idx in range(ncols):
        axes[-1][col_idx].text(0.5, -0.2, labels[col_idx], size=14, ha="center", transform=axes[-1][col_idx].transAxes)

    save_path = f"resolution_model_comparison.{save_format}"
    plt.savefig(save_path, format=save_format, bbox_inches='tight', pad_inches=0.05)
    print(f"\n跨模型与分辨率对比图已成功生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATHS = [
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_002372.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2010_000238.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2007_000904.jpg'
    ] 
    
    # 请确保这两个 fine-tune 后的权重文件存在
    LARGE_WEIGHT = 'latest_vtp_pspnet_large_baseline.pth' 
    SMALL_WEIGHT = 'latest_vtp_pspnet_small_unfreeze.pth'
    
    generate_cross_comparison(TEST_IMAGE_PATHS, LARGE_WEIGHT, SMALL_WEIGHT, save_format='svg')