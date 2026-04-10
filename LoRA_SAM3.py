import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
import math

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
# class PixelAdaptiveLoraLinear(nn.Module):
#     """
#     Wraps a PEFT LoraLinear layer to add a learnable, spatially-adaptive alpha map.
#     Handles both 3D (B, N, C) and 4D (B, H, W, C) inputs robustly.
#     """
#     def __init__(self, lora_layer: LoraLinear, grid_size=32):
#         super().__init__()
#         self.lora_layer = lora_layer
#         self.spatial_alpha = nn.Parameter(torch.ones(1, 1, grid_size, grid_size))

#     def forward(self, x):
#         B, H, W, C = x.shape

#         result_base = self.lora_layer.base_layer(x)
        
#         adapter = self.lora_layer.active_adapter[0] if isinstance(self.lora_layer.active_adapter, list) else self.lora_layer.active_adapter
#         lora_A = self.lora_layer.lora_A[adapter]
#         lora_B = self.lora_layer.lora_B[adapter]
#         dropout = self.lora_layer.lora_dropout[adapter]
#         scaling = self.lora_layer.scaling[adapter]
        
#         x_drop = dropout(x)
#         delta = (x_drop @ lora_A.weight.T) @ lora_B.weight.T
        
#         current_alpha_map = F.interpolate(
#             self.spatial_alpha, 
#             size=(H, W), 
#             mode='bilinear', 
#             align_corners=False
#         ) 

#         alpha_broadcast = current_alpha_map.permute(0, 2, 3, 1) 
#         return result_base + (delta * scaling * alpha_broadcast)


# -------------------------------------------------
# 2. Updated Injection Function
# -------------------------------------------------
def apply_pixel_lora_to_backbone(encoder, lora_config, min_rank=4):
    """
    Applies LoRA using PEFT, then wraps the layers in PixelAdaptiveLoraLinear
    """
    trunk = encoder.vision_backbone.trunk
    num_blocks = len(trunk.blocks)
    base_r = int(lora_config.r)
    print(f"Injecting Pixel-Adaptive LoRA into {num_blocks} blocks...")

    for i, block in enumerate(trunk.blocks):
        # Calculate Rank (Linear decay example)
        r_i = max(min_rank, int(round(base_r * (1 - i / (num_blocks - 1)))))
        
        # Define injector helper
        def inject_and_wrap(module, layer_name):
            target = getattr(module, layer_name, None)
            if isinstance(target, nn.Linear):
                peft_layer = LoraLinear(
                    target, adapter_name="default", r=r_i,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    fan_in_fan_out=False, bias="none"
                )
                # peft_layer = PixelAdaptiveLoraLinear(peft_layer, grid_size=32)
                setattr(module, layer_name, peft_layer)

        if hasattr(block, "attn"):
            inject_and_wrap(block.attn, "qkv")
            inject_and_wrap(block.attn, "proj")

        if hasattr(block, "mlp") and hasattr(block.mlp, "layers"):
            for idx, layer in enumerate(block.mlp.layers):
                if isinstance(layer, nn.Linear):
                    target = layer
                    peft_layer = LoraLinear(
                        target, adapter_name="default", r=r_i,
                        lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        fan_in_fan_out=False, bias="none"
                    )
                    # peft_layer = PixelAdaptiveLoraLinear(peft_layer, grid_size=32)
                    block.mlp.layers[idx] = peft_layer

# -------------------------------------------------
# Layers: Attention Gates & Up Blocks
# -------------------------------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

    def forward(self, g, x):
        g1 = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g1)
        x1 = self.W_x(x)
        psi = self.gelu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn_gate = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        x_skip_clean = self.attn_gate(g=x_up, x=x_skip)
        return self.conv(torch.cat([x_skip_clean, x_up], dim=1))
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        return self.conv(torch.cat([x_skip, x_up], dim=1))

# -------------------------------------------------
# Main Model
# -------------------------------------------------
class LoRA_SAM3(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.backbone
        
        # 1. Cleanup
        if hasattr(self.encoder, 'language_backbone'):
            del self.encoder.language_backbone
            del self.encoder.act_ckpt_whole_language_backbone

        # 2. Freeze Backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3. Apply LoRA
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                 task_type="FEATURE_EXTRACTION", target_modules=["qkv", "proj", "mlp"], use_rslora=True)
        apply_pixel_lora_to_backbone(self.encoder, lora_config, min_rank=4)

        # Layers
        self.bottleneck_conv = DoubleConv(512, 256) 
        self.up1 = AttentionUp(in_channels=256, out_channels=256, skip_channels=256)
        self.up2 = AttentionUp(in_channels=256, out_channels=256, skip_channels=256)

        # self.up1 = Up(in_channels=256, out_channels=256, skip_channels=256)
        # self.up2 = Up(in_channels=256, out_channels=256, skip_channels=256)

        self.head_aux = nn.Conv2d(256, 1, 1)
        self.head_main = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        target_size = x.shape[2:]
        
        # --- Encoder ---
        output = self.encoder.forward_image(x)
        s1, s2, s3 = output["backbone_fpn"][:3]
        v_feat = output["vision_features"]

        btl = torch.cat([s3, v_feat], dim=1)
        btl = self.bottleneck_conv(btl)

        x144 = self.up1(btl, s2)  
        out_aux = self.head_aux(x144)
        out_aux = F.interpolate(out_aux, size=target_size, mode="bilinear", align_corners=False)
        
        x288 = self.up2(x144, s1)
        out_main = self.head_main(x288)
        out_main = F.interpolate(out_main, size=target_size, mode="bilinear", align_corners=False)

        return out_aux, out_main