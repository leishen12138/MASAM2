import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from sam2.build_sam import build_sam2
# from ptflops import get_model_complexity_info


class FrequencyDomainEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_freq_mask = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.high_freq_mask = nn.Parameter(torch.ones(1, channels, 1, 1))

        self.high_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.low_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()  # 生成空间注意力
        )

    def forward(self, x):
        _, _, h, w = x.shape

        x_fft = fft.fftn(x, dim=(-2, -1))
        x_fft_shifted = fft.fftshift(x_fft)

        low_freq = x_fft_shifted * self.low_freq_mask
        high_freq = x_fft_shifted * self.high_freq_mask

        low_freq = fft.ifftn(fft.ifftshift(low_freq), dim=(-2, -1)).real
        high_freq = fft.ifftn(fft.ifftshift(high_freq), dim=(-2, -1)).real

        high_feat = self.high_path(high_freq)

        low_feat = self.low_path(low_freq)
        low_feat = x * low_feat  # 低频信息加权

        return x + high_feat + low_feat


class CrossModalInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.modalA_conv = nn.Conv2d(channels, channels, 1)

        self.modalB_conv = nn.Conv2d(channels, channels, 1)

        self.cross_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x_a, x_b):
        feat_a = self.modalA_conv(x_a)  # [B, C, H, W]
        feat_b = self.modalB_conv(x_b)

        b, c, h, w = feat_a.shape
        feat_a_flat = feat_a.view(b, c, -1).permute(0, 2, 1)
        feat_b_flat = feat_b.view(b, c, -1).permute(0, 2, 1)

        attn_out, _ = self.cross_attn(
            query=feat_a_flat,
            key=feat_b_flat,
            value=feat_b_flat
        )
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)

        gate_input = torch.cat([feat_a, attn_out], dim=1)
        gate = self.fusion_gate(gate_input.mean(dim=(2, 3)))  # [B, C]
        gate = gate.view(b, c, 1, 1)

        return x_a + gate * attn_out


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def structure_loss(self, pred, target):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * target) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def frequency_loss(self, pred, target):
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=torch.float32, device=pred.device)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        pred_lap = F.conv2d(pred, lap_kernel, padding=1)
        target_lap = F.conv2d(target, lap_kernel, padding=1)
        return F.l1_loss(pred_lap, target_lap)

    def forward(self, pred, target):
        loss_struct = self.structure_loss(pred, target)
        loss_freq = self.frequency_loss(torch.sigmoid(pred), target)
        return loss_struct + 0.2 * loss_freq  # 加权组合



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        device = torch.device("cuda")
        self.block = blk.to(device)
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32).to(device),
            nn.GELU(),
            nn.Linear(32, dim).to(device),
            nn.GELU()
        )

    def forward(self, x):
        if x.device != next(self.prompt_learn.parameters()).device:
            x = x.to(next(self.prompt_learn.parameters()).device)
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        device = torch.device("cuda")
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False).to(device)
        self.bn = nn.BatchNorm2d(out_planes).to(device)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.freq_enhence = FrequencyDomainEnhancement(out_channel)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        x = self.freq_enhence(x)
        return x


class MASAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.cross_modal = CrossModalInteraction(64)
        self.to(torch.device('cuda'))
        for module in self.modules():
            if hasattr(module, '_parameters'):
                for name, param in module._parameters.items():
                    if param is not None and not param.is_cuda:
                        module._parameters[name] = param.cuda()

    def forward(self, x_rgb, x_depth=None):
        x1, x2, x3, x4 = self.encoder(x_rgb)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        if x_depth is not None and hasattr(self, 'cross_modal'):
            with torch.no_grad():  
                d1, _, _, _ = self.encoder(x_depth)  
                d1 = self.rfb1(d1)                   
                x1 = self.cross_modal(x1, d1)            
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


# if __name__ == "__main__":
#     with torch.no_grad():
#         model = SAM2UNet().cuda()
#         x = torch.randn(1, 3, 352, 352).cuda()
#         out, out1, out2 = model(x)
#         print(out.shape, out1.shape, out2.shape)
if __name__ == "__main__":
    model = SAM2UNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # out, out1, out2 = model(x)
    # print(out.shape, out1.shape, out2.shape)
    if hasattr(model, 'cross_modal'):
        model.cross_modal = nn.Identity()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # macs, params = get_model_complexity_info(model, (3, 352, 352), as_strings=False, print_per_layer_stat=False)
    # gflops = macs * 2 / 1e9  
    # print(f"\n总计算量: {gflops:.2f} GFLOPs (输入尺寸 352x352)")  

    model.eval()

    for _ in range(10):
        _ = model(input_tensor)

    repetitions = 100
    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(input_tensor)
        end.record()
        torch.cuda.synchronize()

        timings.append(start.elapsed_time(end))

    avg_time = sum(timings) / repetitions
    print(f"平均推理时间: {avg_time:.2f} ms (Batch=1)")