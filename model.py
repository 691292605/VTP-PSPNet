import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from VTP_main.vtp.models.vtp_hf.modeling_vtp import VTPModel 
from VTP_main.vtp.models.vtp_hf.configuration_vtp import VTPConfig

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)

class VTPBackbone(nn.Module):
    def __init__(self, config_path="./backbone_weight_large/config.json", weight_path="./backbone_weight_large/model.safetensors"):
        super().__init__()
        config = VTPConfig.from_json_file(config_path)
        self.vtp = VTPModel(config) 
        state_dict = load_file(weight_path, device="cpu")
        self.vtp.load_state_dict(state_dict, strict=False)        
        for param in self.vtp.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.vtp.get_intermediate_layers_feature(x, n=4, reshape=True, norm=True)
        feature_map = torch.stack(features, dim=0).mean(dim=0)
        return feature_map
    
class PPM(nn.Module):
    def __init__(self, in_dim=384, reduction_dim=128, bins=(1, 2, 3, 6)):

        super(PPM, self).__init__()
        self.features = nn.ModuleList()
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            feat = f(x)
            upsampled = F.interpolate(feat, x_size[2:], mode='bilinear', align_corners=True)
            out.append(upsampled)
        return torch.cat(out, 1)

class VTP_PSPNet(nn.Module):
    def __init__(self, num_classes=21, use_dropout=False, use_adapter=False,
                 config_path="./backbone_weight_large/config.json", 
                 weight_path="./backbone_weight_large/model.safetensors"):
        super().__init__()
        self.backbone = VTPBackbone(config_path=config_path, weight_path=weight_path) 
        embed_dim = self.backbone.vtp.config.vision_embed_dim
        self.ppm = PPM(in_dim=embed_dim, reduction_dim=128, bins=(1, 2, 3, 6))
        concat_channels = embed_dim
        pspnet_in_channels = concat_channels
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.spatial_dropout = nn.Dropout2d(p=0.15)
        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = FeatureAdapter(in_channels=concat_channels, out_channels=concat_channels)
            pspnet_in_channels = concat_channels
        ppm_out_channels = embed_dim + 128 * 4
        self.cls_head = nn.Sequential(
            nn.Conv2d(ppm_out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_dropout:
            feat = self.spatial_dropout(feat)
        if self.use_adapter:
            feat = self.adapter(feat)
        ppm_out = self.ppm(feat)
        logits = self.cls_head(ppm_out)
        out = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    backbone = VTPBackbone()
    for name, param in backbone.vtp.named_parameters():
        print(name)

    """ dummy_input = torch.randn(2, 3, 1024, 1024)
    backbone.eval()

    with torch.no_grad():
        final_feature = backbone(dummy_input)
    print(f"输出特征图的维度是: {final_feature.shape}")
    # 预期输出: torch.Size([2, 384, 64, 64]) 或 [2, 768, 64, 64]
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    model = VTP_PSPNet(num_classes=21).to(device)

    block_names = [n for n, p in model.backbone.vtp.named_parameters() if "blocks." in n]
    max_block_idx = max([int(n.split("blocks.")[1].split(".")[0]) for n in block_names])
    print(f"检测到 VTP 最大 Block 索引为: {max_block_idx}")
    print(f"模型共有 {max_block_idx + 1} 层 (0 到 {max_block_idx})")

    model.eval()
    with torch.no_grad():
        final_output = model(dummy_input)
    if final_output.shape == (1, 21, 512, 512):
        print("测试成功，可以开始训练。")
    else:
        print("维度不匹配，请检查上述输出记录。") """