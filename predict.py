import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

    print("正在加载对比模型 (FCN & DeepLab)...")
    fcn_model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT).to(device).eval()
    deeplab_model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_imgs = len(image_paths)
    ncols = 5  

    fig, axes = plt.subplots(
        nrows=num_imgs, 
        ncols=ncols, 
        figsize=(15, 3 * num_imgs), 
        gridspec_kw={'wspace': 0.05, 'hspace': 0.1} 
    )
    
    if num_imgs == 1:
        axes = [axes]

    for i, image_path in enumerate(image_paths):
        print(f"正在处理第 {i+1}/{num_imgs} 张图片...")

        # (a) 原图：严格按照你要求的统一缩放到 1024 (展示用)
        original_img = Image.open(image_path).convert('RGB')
        axes[i][0].imshow(original_img.resize((1024, 1024))) 
        axes[i][0].axis('off')

        # (b) Ground Truth
        gt_path = image_path.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path)
            gt_mask = np.array(gt_img.resize((1024, 1024), Image.NEAREST))
            axes[i][1].imshow(decode_segmap(gt_mask))
        else:
            axes[i][1].text(0.5, 0.5, "GT Not Found", ha='center', va='center')
        axes[i][1].axis('off')

        input_tensor = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # (c) FCN
            out_fcn = fcn_model(input_tensor)['out'][0]
            mask_fcn = torch.argmax(out_fcn, dim=0).cpu().numpy()
            axes[i][2].imshow(decode_segmap(mask_fcn))
            axes[i][2].axis('off')

            # (d) DeepLab
            out_deeplab = deeplab_model(input_tensor)['out'][0]
            mask_deeplab = torch.argmax(out_deeplab, dim=0).cpu().numpy()
            axes[i][3].imshow(decode_segmap(mask_deeplab))
            axes[i][3].axis('off')

            # (e) VTP+PSPNet (本文模型)
            out_ours = vtp_model(input_tensor)
            mask_ours = torch.argmax(out_ours.squeeze(0), dim=0).cpu().numpy()
            axes[i][4].imshow(decode_segmap(mask_ours))
            axes[i][4].axis('off')

    # 严格保持你的底部标注文本
    labels = ["(a) Image", "(b) Ground Truth", "(c) FCN", "(d) DeepLab", "(e) VTP-PSPNet Baseline"]
    for col_idx in range(ncols):
        axes[-1][col_idx].text(0.5, -0.2, labels[col_idx], size=14, ha="center", transform=axes[-1][col_idx].transAxes)

    save_path = f"vtp_comparison_figure.{save_format}"
    plt.savefig(save_path, format=save_format, bbox_inches='tight', pad_inches=0.05)
    
    print(f"\n对比图已成功生成并保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATHS = [
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2007_004866.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2008_000359.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_004993.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2010_002418.jpg', 
        r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg'   
    ] 
    
    # 指向你训练好的最佳权重文件
    WEIGHT_PATH = 'latest_vtp_pspnet_small_unfreeze.pth' 
    visualize_vtp_comparison(TEST_IMAGE_PATHS, WEIGHT_PATH)