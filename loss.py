import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super(CE_DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, target):
        inputs = inputs.float()
        
        ce = self.ce_loss(inputs, target)
        pred = F.softmax(inputs, dim=1) 
        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.num_classes)
        target = target.view(-1)
        
        valid_mask = (target != self.ignore_index)
        pred = pred[valid_mask]
        target = target[valid_mask]  

        if target.numel() == 0:
            return ce
        target = torch.clamp(target, 0, self.num_classes - 1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        smooth = 1e-5
        intersection = torch.sum(pred * target_one_hot, dim=0)
        union = torch.sum(pred, dim=0) + torch.sum(target_one_hot, dim=0)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_foreground = dice_score[1:]
        dice = 1.0 - torch.mean(dice_foreground)
        return ce + dice

class Focal_DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, gamma=2.0):
        super(Focal_DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.ce_none = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, target):
        inputs = inputs.float()
        
        invalid_mask = (target >= self.num_classes) & (target != self.ignore_index)
        target[invalid_mask] = self.ignore_index
        
        ce_loss_unreduced = self.ce_none(inputs, target) 
        pt = torch.exp(-ce_loss_unreduced)              
        focal_loss_map = ((1 - pt) ** self.gamma) * ce_loss_unreduced
        
        valid_mask_focal = (target != self.ignore_index)
        if valid_mask_focal.sum() > 0:
            focal_loss = focal_loss_map[valid_mask_focal].mean()
        else:
            focal_loss = focal_loss_map.sum() 
            
        pred = F.softmax(inputs, dim=1) 
        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.num_classes)
        target_flat = target.view(-1)
        
        valid_mask = (target_flat != self.ignore_index)
        pred = pred[valid_mask]
        target_flat = target_flat[valid_mask]  

        if target_flat.numel() == 0:
            return focal_loss

        target_flat = torch.clamp(target_flat, 0, self.num_classes - 1)
        target_one_hot = F.one_hot(target_flat, num_classes=self.num_classes).float()
        smooth = 1e-5
        intersection = torch.sum(pred * target_one_hot, dim=0)
        union = torch.sum(pred, dim=0) + torch.sum(target_one_hot, dim=0)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_foreground = dice_score[1:]
        dice = 1.0 - torch.mean(dice_foreground)
        
        return focal_loss + dice