import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms
from model import VTPBackbone 

def get_pca_map(features, h_target, w_target):
    """通用的 PCA 降维并归一化函数"""
    B, C, H, W = features.shape
    feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    
    feat_norm = torch.nn.functional.normalize(feat_flat, p=2, dim=1)

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(feat_norm.cpu().numpy())

    for i in range(3):
        low, high = np.percentile(pca_results[:, i], [1, 99]) 
        pca_results[:, i] = np.clip(pca_results[:, i], low, high)
        pca_results[:, i] = (pca_results[:, i] - low) / (high - low + 1e-8)

    pca_img = pca_results.reshape(H, W, 3)
    pca_img_resized = cv2.resize(pca_img, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
    return pca_img_resized

def run_vtp_ablation_pca(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((2048, 2048)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    model = VTPBackbone().to(device)
    model.eval()

    print("正在提取特征并进行 PCA 降维...")
    with torch.no_grad():
        feats_n4 = model.vtp.get_intermediate_layers_feature(input_tensor, n=4, reshape=True, norm=True)
        feat_mean4 = torch.stack(feats_n4).mean(dim=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['(a) Original Image', '(b) PCA Feature Map']
    
    axes[0].imshow(np.array(original_img.resize((2048, 2048))))
    
    pca_map = get_pca_map(feat_mean4, 2048, 2048)
    axes[1].imshow(pca_map)

    for ax in axes:
        ax.axis('off')
    
    axes[0].text(0.5, -0.02, titles[0], transform=axes[0].transAxes, ha='center', va='top', fontsize=14)
    axes[1].text(0.5, -0.02, titles[1], transform=axes[1].transAxes, ha='center', va='top', fontsize=14)

    plt.tight_layout()
    
    save_path = 'vtp_feature_pca_result.svg'
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    IMG_PATH = r'D:/lab/dinov3_pspnet/data/VOCdevkit/VOC2012/JPEGImages/2009_004993.jpg'
    run_vtp_ablation_pca(IMG_PATH)