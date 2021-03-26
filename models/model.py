import sys

sys.path.append('/home/ljc/works/fast-reid/')

import torch
import torch.nn as nn
from fastreid.modeling.backbones.resnet import build_resnet_backbone

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class ReidNet(nn.Module):
    '''
    Re-id backbone. Consist of:
        1. Resnet50
        2. GAP
        3. BN
        4. L2-normalization
    Output: (batchsize, 2048) features.
    '''
    def __init__(self, cfg):
        '''
        Args:
            cfg: YML config file of model configurations.
        
        Returns:
            A ReidNet instance.
        '''
        super(ReidNet, self).__init__()
        self.cfg = cfg
        self.backbone = self._init_backbone()
        self.head = self._init_head()

        # bottleneck
        self.bottleneck = nn.BatchNorm1d(self.cfg.MODEL.BACKBONE.FEAT_DIM)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
    
    def _init_backbone(self):
        return build_resnet_backbone(self.cfg)

    def _init_head(self):
        head = nn.Sequential(
            GAPLayer(),
            # nn.BatchNorm2d(self.cfg.MODEL.BACKBONE.FEAT_DIM)
        )
        return head

    def l2_normalize(self, feat):
        norm = torch.sqrt(torch.sum(torch.square(feat), dim=1))
        return torch.div(feat.t(), norm.t()).t()

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.head(feat).squeeze()

        # bottleneck
        feat = self.bottleneck(feat)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        # feat = feat.view((-1, self.cfg.MODEL.BACKBONE.FEAT_DIM)) # [bsize, 2048, 1, 1] -> [bsize, 2048]
        # feat = self.l2_normalize(feat) # l2 normalization
        return feat

class GAPLayer(nn.Module):
    def __init__(self):
        super(GAPLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # bsize, ch, h, w = x.shape
        # x = nn.AvgPool2d((h, w))(x)
        x = self.pool(x)
        return x

if __name__ == "__main__":
    import numpy as np
    from fastreid.config.config import CfgNode
    with open('backbone.yml', 'r') as f:
        cfg = CfgNode.load_cfg(f)
    x = torch.randn((64, 3, 256, 256))
    model = ReidNet(cfg)
    y = model(x)
    print(y.shape)