# Общие frozen encoder для экспериментов и offline-сервиса.
import torch
from torch import nn
from torchvision import transforms
from .poi_baseline import sha256_file,create_encoder


class DinoAttention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(384,1152);self.proj=nn.Linear(384,384)
    def forward(self,x):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,6,64).permute(2,0,3,1,4).unbind(0)
        return self.proj(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,c))


class DinoScale(nn.Module):
    def __init__(self):super().__init__();self.gamma=nn.Parameter(torch.ones(384))
    def forward(self,x):return x*self.gamma


class DinoMlp(nn.Module):
    def __init__(self):super().__init__();self.fc1=nn.Linear(384,1536);self.fc2=nn.Linear(1536,384)
    def forward(self,x):return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(384,eps=1e-6);self.norm2=nn.LayerNorm(384,eps=1e-6)
        self.attn=DinoAttention();self.mlp=DinoMlp();self.ls1=DinoScale();self.ls2=DinoScale()
    def forward(self,x):
        x=x+self.ls1(self.attn(self.norm1(x)))
        return x+self.ls2(self.mlp(self.norm2(x)))


class DinoPatch(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(3,384,14,14)
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)


class DinoSmall(nn.Module):
    """Eval-only ViT-S/14 для официального non-register DINOv2 checkpoint.

    Имена параметров совместимы с upstream; strict=True исключает частичную загрузку.
    Interpolation offset=0.1 и bicubic соответствуют исходной реализации DINOv2.
    """
    def __init__(self):
        super().__init__();self.patch_embed=DinoPatch()
        self.cls_token=nn.Parameter(torch.zeros(1,1,384));self.mask_token=nn.Parameter(torch.zeros(1,384))
        self.pos_embed=nn.Parameter(torch.zeros(1,1370,384))
        self.blocks=nn.ModuleList([DinoBlock() for _ in range(12)])
        self.norm=nn.LayerNorm(384,eps=1e-6)
    def forward(self,x):
        h,w=x.shape[-2:];x=self.patch_embed(x)
        x=torch.cat([self.cls_token.expand(x.shape[0],-1,-1),x],dim=1)
        if h==518 and w==518:position=self.pos_embed
        else:
            patches=self.pos_embed[:,1:].reshape(1,37,37,384).permute(0,3,1,2)
            patches=torch.nn.functional.interpolate(patches,scale_factor=((h//14+.1)/37,(w//14+.1)/37),mode='bicubic',align_corners=False)
            position=torch.cat([self.pos_embed[:,:1],patches.permute(0,2,3,1).reshape(1,-1,384)],dim=1)
        x=x+position
        for block in self.blocks:x=block(x)
        return self.norm(x)[:,0]


ENCODERS={
    'mobilenetv3_small':{'file':'mobilenet_v3_small-047dcff4.pth','url':'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth','sha_prefix':'047dcff4','dimension':576},
    'mobilenetv3_large':{'file':'mobilenet_v3_large-5c1a4163.pth','url':'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth','sha_prefix':'5c1a4163','dimension':960},
    'dinov2_small':{'file':'dinov2_vits14_pretrain.pth','url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth','sha_prefix':'','dimension':384},
}


def load_encoder(name,weights):
    from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
    digest=sha256_file(weights)
    if not digest.startswith(ENCODERS[name]['sha_prefix']):raise ValueError('SHA checkpoint не совпадает с официальным именем')
    if name=='mobilenetv3_small':return create_encoder(weights)
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if name=='mobilenetv3_large':
        model=mobilenet_v3_large(weights=None);model.load_state_dict(state,strict=True);model.classifier=nn.Identity()
        transform=MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    else:
        model=DinoSmall();model.load_state_dict(state,strict=True)
        transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return model.eval().requires_grad_(False),transform


