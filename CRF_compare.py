import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

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
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def apply_dense_crf(original_image, predicted_probs, num_classes=21):
    """
    使用 DenseCRF 优化语义分割的边缘。
    """
    C, H, W = predicted_probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    U = unary_from_softmax(predicted_probs)
    d.setUnaryEnergy(U)
    img_np = np.ascontiguousarray(original_image)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(20, 20, 20), rgbim=img_np, compat=5)
    d.addPairwiseGaussian(sxy=(7, 7), compat=3)
    Q = d.inference(5)
    refined_mask = np.argmax(Q, axis=0).reshape((H, W))
    return refined_mask

def visualize_vtp_comparison(image_paths, model_weight_path, save_format='svg'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在加载 VTP+PSPNet 模型...")
    vtp_model = VTP_PSPNet(num_classes=21).to(device)
    ckpt = torch.load(model_weight_path, map_location=device)
    if 'model_state_dict' in ckpt:
        vtp_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        vtp_model.load_state_dict(ckpt, strict=False)
    vtp_model.eval()

    # 预处理转换
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_imgs = len(image_paths)
    ncols = 4  # 调整为 4 列：原图, GT, VTP(Raw), VTP(w/ CRF)

    fig, axes = plt.subplots(
        nrows=num_imgs, 
        ncols=ncols, 
        figsize=(12, 3 * num_imgs), # 稍微收窄宽度
        gridspec_kw={'wspace': 0.05, 'hspace': 0.1} 
    )
    
    if num_imgs == 1:
        axes = [axes]

    for i, image_path in enumerate(image_paths):
        print(f"正在处理第 {i+1}/{num_imgs} 张图片...")

        # 图像准备
        original_img = Image.open(image_path).convert('RGB')
        resized_img = original_img.resize((512, 512))
        resized_img_np = np.array(resized_img)
        
        # (a) 原图
        axes[i][0].imshow(resized_img_np) 
        axes[i][0].axis('off')

        # (b) Ground Truth
        gt_path = image_path.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path)
            gt_mask = np.array(gt_img.resize((512, 512), Image.NEAREST))
            axes[i][1].imshow(decode_segmap(gt_mask))
        else:
            axes[i][1].text(0.5, 0.5, "GT Not Found", ha='center', va='center')
        axes[i][1].axis('off')

        input_tensor = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_ours = vtp_model(input_tensor) # shape: (1, 21, 512, 512)
            
            mask_raw = torch.argmax(out_ours.squeeze(0), dim=0).cpu().numpy()
            axes[i][2].imshow(decode_segmap(mask_raw))
            axes[i][2].axis('off')

            probs_ours = torch.softmax(out_ours.squeeze(0), dim=0).cpu().numpy()
            mask_ours_crf = apply_dense_crf(resized_img_np, probs_ours, num_classes=21)
            axes[i][3].imshow(decode_segmap(mask_ours_crf))
            axes[i][3].axis('off')

    labels = ["(a) Image", "(b) Ground Truth", "(c) VTP-PSPNet Baseline", "(d) VTP-PSPNet+CRF"]
    for col_idx in range(ncols):
        axes[-1][col_idx].text(0.5, -0.2, labels[col_idx], size=14, ha="center", transform=axes[-1][col_idx].transAxes)

    save_path = f"vtp_crf_ablation.{save_format}"
    plt.savefig(save_path, format=save_format, bbox_inches='tight', pad_inches=0.05)
    print(f"\n消融对比图已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATHS = [
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_003378.jpg',
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_001991.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_002265.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2010_001646.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2007_002619.jpg'   
    ] 
    # 指向你训练好的最佳权重文件
    WEIGHT_PATH = 'latest_vtp_pspnet_large_baseline.pth' 
    visualize_vtp_comparison(TEST_IMAGE_PATHS, WEIGHT_PATH)