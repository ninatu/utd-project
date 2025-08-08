import os
import logging
from collections import OrderedDict
import torch
from torch import nn

from third_party.internvid.viclip.viclip_vision import VisionTransformer, _MODELS, load_state_dict

logger = logging.getLogger(__name__)


def load_backbone(args):
    vclip = ViCLIP(size=args.model, pretrain=args.backbone_checkpoint, vision_encoder_checkpoint_num=0)
    return vclip


def clip_joint_b16_backbone(
        pretrained=False, input_resolution=224, kernel_size=1,
        center=True, num_frames=8, drop_path=0., checkpoint_num=0,
        dropout=0.,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=None,
        kernel_size=kernel_size, num_frames=num_frames,
        drop_path=drop_path, checkpoint_num=checkpoint_num,
        dropout=dropout,
    )
    # raise NotImplementedError
    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-B/16"

        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_backbone(
        pretrained=False, input_resolution=224, kernel_size=1,
        center=True, num_frames=8, drop_path=0., checkpoint_num=0,
        dropout=0.,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=None,
        kernel_size=kernel_size, num_frames=num_frames,
        drop_path=drop_path, checkpoint_num=checkpoint_num,
        dropout=dropout,
    )

    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-L/14"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


class ViCLIP(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self,  
                 size='l',
                 vision_encoder_checkpoint_num=24,
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth")
                 ):
        super(ViCLIP, self).__init__()

        if size.lower() == 'l':
            self.vision_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.vision_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")
    
        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = vision_encoder_checkpoint_num
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.text_width = 768 
        self.embed_dim = 768 
        self.masking_prob = 0.9

        # create modules.
        self.vision_encoder = self.build_vision_encoder()

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu')['model']

            all_keys = list(state_dict.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('text_encoder.'):
                    pass
                else:
                    new_dict[key] = state_dict[key]
            missing_keys, unexpected_keys = self.load_state_dict(new_dict, strict=False)
            # print(missing_keys, unexpected_keys, flush=True)
            assert missing_keys == []
            assert unexpected_keys == ['temp', 'vision_encoder.proj']

    def get_num_layers(self):
        return self.vision_encoder.get_num_layers()

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )

        return ret

    def forward(self, image):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        vision_embeds = self.encode_vision(image)

        return vision_embeds

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        # if image.ndim == 5:
        #     image = image.permute(0, 2, 1, 3, 4).contiguous()
        # else:
        #     image = image.unsqueeze(2)

        # if not test and self.masking_prob > 0.0:
        #     return self.vision_encoder(
        #         image, masking_prob=self.masking_prob
        #     )

        return self.vision_encoder(image)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            self.embed_dim = 1024

            vision_encoder = clip_joint_l14_backbone(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        elif encoder_name == "vit_b16":
            self.embed_dim = 768

            vision_encoder = clip_joint_b16_backbone(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")
            
        return vision_encoder
