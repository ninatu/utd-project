import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class FrozenBackboneClassifier(nn.Module):
    def __init__(self,
                 backbone,
                 num_classes=1000,
                 fc_drop=0.,
                 init_scale=0.0
                 ):
        super().__init__()

        # set default
        qkv_bias = True
        qk_scale = None
        attn_drop_rate = 0.
        norm_layer = nn.LayerNorm

        embed_dim = backbone.embed_dim
        num_heads_dict = {
            1280: 16,
            1024: 16,
            768: 12,
            576: 9
        }
        mlp_ratio = 4

        num_heads = num_heads_dict[embed_dim]

        self.norm1 = norm_layer(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, bias=qkv_bias,
                                                dropout=attn_drop_rate)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.fc_norm = norm_layer(embed_dim)
        self.fc_drop = nn.Dropout(fc_drop)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)

        self.apply(_init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

        # assign after weight init, otherwise the backbone weights are lost
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        with torch.set_grad_enabled(False):
            self.backbone.eval()
            x = self.backbone(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # pooler
        x = self.norm1(x)
        cls_token = self.norm1(cls_token)
        x = cls_token + self.attn(cls_token, x, x)[0]
        x = x + self.mlp(self.norm2(x))

        # head
        x = self.fc_norm(x)
        x = self.fc_drop(x)
        x = x.squeeze(1)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
