import torch
from torch import nn
from types import SimpleNamespace
from einops import rearrange
from torchvision.models import resnet101, ResNet101_Weights
from collections import OrderedDict


class DINOv2NetMultiScale(nn.Module):
    """
    A wrapper around a pre-trained DINOv2 network (https://github.com/facebookresearch/dinov2)
    """
    def __init__(self, cfg):
        super(DINOv2NetMultiScale, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_resize_factor = cfg.MODEL.FEATURE_RESIZE_FACTOR

        if isinstance(cfg.MODEL.MULTISCALE, int):
            self.blocks_ids =(23 - torch.arange(cfg.MODEL.MULTISCALE)).tolist()
        elif isinstance(cfg.MODEL.MULTISCALE, list):
            self.blocks_ids = cfg.MODEL.MULTISCALE
        else:
            raise TypeError

        self.finetune = cfg.MODEL.FINETUNE

        self.patch_sz = cfg.MODEL.PATCH_SIZE
        assert self.patch_sz == 14, "MODEL.PATCH_SIZE has to be set to 14 for base DINOv2 model!"

        # load dino model 
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', cfg.MODEL.ARCH).to(self.device)

        if not self.finetune:
            # froze parameters
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.dinov2.eval()

    def forward(self, x, norm=True):
        if not self.finetune:
            with torch.no_grad():
                # list of [B (H W) C]
                out_ms = self.dinov2.get_intermediate_layers(x, n=self.blocks_ids, norm=norm)
        else:
            out_ms = self.dinov2.get_intermediate_layers(x, n=self.blocks_ids, norm=norm)

        return out_ms

class  DINOv2NetMultiScaleBaseline(DINOv2NetMultiScale):
    def __init__(self, cfg):
        super(DINOv2NetMultiScaleBaseline, self).__init__(cfg)

        num_layers = len(self.blocks_ids)
        self.decoder = torch.nn.Sequential(
                nn.Conv2d(num_layers*cfg.MODEL.EMB_SIZE, num_layers*cfg.MODEL.EMB_SIZE, kernel_size=1, stride=1),
                nn.GELU(), 
                nn.Conv2d(num_layers*cfg.MODEL.EMB_SIZE, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1)
            )

    def forward(self, x):
        # list of [B (H W) C]
        emb_list = DINOv2NetMultiScale.forward(self, x, norm=True)
        emb = rearrange(torch.cat(emb_list, dim=-1), "b (h w) c -> b c h w", h=int(x.shape[2]/self.patch_sz))
        emb = torch.nn.functional.interpolate(emb, scale_factor = self.feature_resize_factor, mode="bilinear")
        
        # [B, num_classes, xH, xW]
        logits_embshape = self.decoder(emb)
        logits = torch.nn.functional.interpolate(logits_embshape, size=x.shape[-2:], mode="bilinear")

        logits = rearrange(logits, "b c xh xw -> b xh xw c")
        emb = rearrange(emb, "b c ph pw -> b ph pw c")
        logits_embshape = rearrange(logits_embshape, "b c ph pw -> b ph pw c")
        return SimpleNamespace(logits = logits, emb = emb, logits_embshape=logits_embshape)


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

# ============================================================================
# most of the supporting code was taken from https://github.com/valeoai/obsnet
# ============================================================================

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    return nn.BatchNorm2d(in_channels)

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x, **kwargs):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class DINOv2NetMultiScaleDeepLabDecoder(DINOv2NetMultiScale):
    def __init__(self, cfg):
        super(DINOv2NetMultiScaleDeepLabDecoder, self).__init__(cfg)
        self.emb_size = cfg.MODEL.EMB_SIZE

        num_layers = len(self.blocks_ids)
        assert num_layers == 4, "Hard coded number of blocks to 4"

        self.aspp = _AtrousSpatialPyramidPoolingModule(2*cfg.MODEL.EMB_SIZE, 256, output_stride=8)

        self.bot_fine = nn.Conv2d(2*cfg.MODEL.EMB_SIZE, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.MODEL.NUM_CLASSES, kernel_size=1, bias=False))

        self.initialize_weights(self.final)

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x, target=None):
        # list of [B (H W) C]
        emb_list = DINOv2NetMultiScale.forward(self, x, norm=True)
        emb = rearrange(torch.cat(emb_list, dim=-1), "b (h w) c -> b c h w", h=int(x.shape[2]/self.patch_sz))
        emb = torch.nn.functional.interpolate(emb, scale_factor = self.feature_resize_factor, mode="bilinear")

        xd = self.aspp(emb[:, -2*self.emb_size:, ...])
        dec0_0 = self.bot_aspp(xd)
        dec0_1 = self.bot_fine(emb[:, :2*self.emb_size, ...])
        dec0 = torch.cat([dec0_0, dec0_1], dim=1)
        logits_embshape = self.final(dec0)

        logits = torch.nn.functional.interpolate(logits_embshape, size=x.shape[-2:], mode="bilinear")

        logits = rearrange(logits, "b c xh xw -> b xh xw c")
        logits_embshape = rearrange(logits_embshape, "b c ph pw -> b ph pw c")
        emb = rearrange(emb, "b c ph pw -> b ph pw c")

        return SimpleNamespace(logits = logits, emb = emb, logits_embshape=logits_embshape)

class DINOv2DeepLab(DINOv2NetMultiScale):
    def __init__(self, cfg):
        super(DINOv2DeepLab, self).__init__(cfg)
        self.emb_size = cfg.MODEL.EMB_SIZE

        self.cfg = cfg
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)

        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.MODEL.NUM_CLASSES, kernel_size=1, bias=False))

        self.initialize_weights(self.final)

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x, target=None):
        # list of [B (H W) C]
        emb_list = DINOv2NetMultiScale.forward(self, x, norm=True)
        emb = rearrange(torch.cat(emb_list, dim=-1), "b (h w) c -> b c h w", h=int(x.shape[2]/self.patch_sz))
        emb = torch.nn.functional.interpolate(emb, scale_factor = self.feature_resize_factor, mode="bilinear")

        features = self.backbone(x)
        x_lowlevel = features["low_level"]
        x_out = features["out"]

        dec0_up = self.aspp(x_out)
        dec0_up = self.bot_aspp(dec0_up)
        dec0_up = Upsample(dec0_up, x_lowlevel.shape[-2:])
        dec0_fine = self.bot_fine(x_lowlevel)
        dec0 = torch.cat([dec0_up, dec0_fine], dim=1)
        dec1 = self.final(dec0)

        logits = torch.nn.functional.interpolate(dec1, size=x.shape[-2:], mode="bilinear")
        logits = rearrange(logits, "b c xh xw -> b xh xw c")

        logits_embshape = torch.nn.functional.interpolate(dec1, size=emb.shape[-2:], mode="bilinear")
        logits_embshape = rearrange(logits_embshape, "b c ph pw -> b ph pw c")

        emb = rearrange(emb, "b c ph pw -> b ph pw c")

        return SimpleNamespace(logits = logits, emb = emb, logits_embshape=logits_embshape)
