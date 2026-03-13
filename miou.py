import torch
import numpy as np
from tqdm import tqdm
from model import VTP_PSPNet
from data import voc_dataloaders

# PASCAL VOC 2012 的 21 个类别名称（包含背景）
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def fast_hist(a, b, n):
    """
    计算混淆矩阵 (Confusion Matrix)
    a: 真实标签 (Ground Truth)
    b: 预测结果 (Prediction)
    n: 类别数量
    """
    # 提取有效的像素（忽略标签为 255 的边缘区域）
    k = (a >= 0) & (a < n)
    # 利用 numpy 的 bincount 快速计算 1D 数组的频数统计，并重塑为 n*n 的混淆矩阵
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def evaluate_model(weight_path, num_classes=21):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行验证集评估...")

    # 1. 加载模型与权重
    model = VTP_PSPNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    # 2. 获取验证集 DataLoader
    # 这里我们复用 data.py 中的函数，只取 val_loader，批次大小设为 4 或 8（根据你的显存决定）
    _, val_loader = voc_dataloaders(batch_size=4)

    hist = np.zeros((num_classes, num_classes))
    
    print("开始计算 mIoU，这可能需要几分钟时间...")
    pbar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.numpy() 
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for pred, mask in zip(preds, masks):
                hist += fast_hist(mask.flatten(), pred.flatten(), num_classes)

    # 3. 计算评价指标
    # 像素准确率 PA = 对角线元素之和 / 所有元素之和
    acc = np.diag(hist).sum() / hist.sum()
    
    # 每个类别的准确率 = 对角线元素 / 对应行的和
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    
    # 每个类别的 IoU = 交集 / 并集
    # 交集：对角线元素 (diag)
    # 并集：预测为该类的总数 + 真实为该类的总数 - 预测且真实为该类的总数 (即交集)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    
    # mIoU (忽略除以0的无效类别)
    valid = hist.sum(axis=1) > 0  # 确保验证集中确实存在该类别
    mean_iu = np.nanmean(iu[valid])

    print(f"评估完成! 使用的权重: {weight_path}")
    print(f"{'Class Name':<15} | {'IoU (%)':<10}")
    
    for i in range(num_classes):
        if valid[i]:
            print(f"{VOC_CLASSES[i]:<15} | {iu[i] * 100:>5.2f}")
        else:
            print(f"{VOC_CLASSES[i]:<15} | N/A")
            
    print("="*50)
    print(f"Mean IoU (mIoU): {mean_iu * 100:.2f} %")
    print(f"Pixel Acc (PA) : {acc * 100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    WEIGHT_PATH = 'latest_vtp_pspnet_large_baseline.pth'
    evaluate_model(WEIGHT_PATH)