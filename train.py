import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import VTP_PSPNet
from data import voc_dataloaders
from loss import CE_DiceLoss, Focal_DiceLoss


def main():
    RESUME = True  # 设置为 True 以开启断点续传
    CHECKPOINT_PATH = 'checkpoint.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    model = VTP_PSPNet(num_classes=21).to(device)
    print(f"当前使用的设备是: {device}")
    
    Freeze_Epoch = 5         #冻结主干
    Freeze_batch_size = 4   
    Freeze_lr = 1e-3 
    
    UnFreeze_Epoch = 50      #解冻主干
    Unfreeze_batch_size = 1 
    Unfreeze_lr = 1e-4      
    
    Total_Epoch = Freeze_Epoch + UnFreeze_Epoch
    
    model = VTP_PSPNet(num_classes=21).to(device)

    criterion = CE_DiceLoss(num_classes=21, ignore_index=255)
    #criterion = Focal_DiceLoss(num_classes=21, ignore_index=255)
    #criterion = nn.CrossEntropyLoss(ignore_index=255)

    scaler = GradScaler('cuda') 
    best_val_loss = float('inf')

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print(f"检测到检查点，正在从 {CHECKPOINT_PATH} 恢复训练...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"恢复成功！将从第 {start_epoch + 1} 个 Epoch 开始。")
    
    for epoch in range(start_epoch,Total_Epoch):
        if epoch < Freeze_Epoch:
            if epoch == 0 or epoch == start_epoch:
                print("[阶段一] 冻结主干，仅训练 PSPNet 头")
                for param in model.backbone.parameters():
                    param.requires_grad = False
                
                train_loader, val_loader = voc_dataloaders(batch_size=Freeze_batch_size)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Freeze_lr, weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=Freeze_Epoch, eta_min=1e-5)

                if RESUME and os.path.exists(CHECKPOINT_PATH) and 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        else:
            if epoch == Freeze_Epoch or epoch == start_epoch:
                print("[阶段二] 解冻主干，微调整个模型")

                for param in model.backbone.parameters():
                    param.requires_grad = False
                for name, param in model.backbone.vtp.named_parameters():
                    if any(f"trunk.blocks.{i}" in name for i in range(20, 24)) or "trunk.norm" in name:
                        param.requires_grad = True

                        
                train_loader, val_loader = voc_dataloaders(batch_size=Unfreeze_batch_size)
                
                backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
                head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
                model.to(device)
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': Unfreeze_lr * 0.1}, 
                    {'params': head_params, 'lr': Unfreeze_lr}
                ], weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch, eta_min=1e-6)

                if RESUME and os.path.exists(CHECKPOINT_PATH) and epoch == start_epoch:
                    if start_epoch != Freeze_Epoch:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    else:
                        print("[跨阶段初始化]优化器参数组已改变，跳过旧状态，使用全新优化器")
        
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Total_Epoch} [Train]")


        """ print("正在核对模型冻结/解冻状态...")
        trainable_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # 只打印被解冻的层，防止输出太多刷屏
                print(f"解冻 (参与训练): {name} | shape: {param.shape}")
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()


        print(f"冻结参数总量: {frozen_params:,}")
        print(f"可训练参数总量: {trainable_params:,}")
        print(f"可训练比例: {trainable_params / (trainable_params + frozen_params) * 100:.2f}%") """

        
        accumulation_steps = 8 
        optimizer.zero_grad()


        for i, (images, masks) in enumerate(pbar_train):
            images, masks = images.to(device), masks.to(device)
            
            
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                raw_loss = loss.item()
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            pbar_train.set_postfix({'loss': f"{raw_loss:.4f}", 'lr': f"{optimizer.param_groups[-1]['lr']:.6f}"})

        torch.save(model.state_dict(), 'latest_vtp_pspnet.pth')
        print(f"已保存 Epoch {epoch+1} 的最新权重: latest_vtp_pspnet.pth")
            
        scheduler.step()
        
        model.eval()
        torch.cuda.empty_cache()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Total_Epoch} [Val]")
            for images, masks in pbar_val:
                images, masks = images.to(device), masks.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{Total_Epoch}] 验证集 Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_vtp_pspnet.pth')
            print(f"保存最佳权重 (Loss: {best_val_loss:.4f})\n")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        print(f"Checkpoint 已更新至 Epoch {epoch+1}")

if __name__ == "__main__":
    main()