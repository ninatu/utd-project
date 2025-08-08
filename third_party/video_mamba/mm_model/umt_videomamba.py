import logging
import os
import os.path as osp
import torch
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .backbones.bert.tokenization_bert import BertTokenizer
from .backbones.videomamba import build_videomamba
from .backbones.bert.builder import build_bert


logger = logging.getLogger(__name__)


class UMT_VIDEOMAMBA(nn.Module):
    """docstring for UMT"""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super(UMT_VIDEOMAMBA, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

    def get_itm_score(self, encoder_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask):
        output = self.text_encoder(encoder_embeds=encoder_embeds,
                                    attention_mask=attention_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    return_dict=True,
                                    mode="fusion",
                                    )

        score = self.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        return score

    def get_vision_embedding_and_features(self, video):
        vision_embeds, pooled_vision_embeds = self.encode_vision(video, test=True)
        pooled_vision_embeds = self.vision_proj(pooled_vision_embeds).mean(dim=1)
        return vision_embeds, pooled_vision_embeds

    def get_text_embedding_and_features(self, text):
        text_embeds, pooled_text_embeds = self.encode_text(text)
        pooled_text_embeds = self.text_proj(pooled_text_embeds)
        return text_embeds, pooled_text_embeds

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal=self.config.model.vision_encoder.keep_temporal
        vision_embeds, pooled_vision_embeds, _ = self.vision_encoder(
            image, None, use_image, keep_temporal,
        )
        return vision_embeds, pooled_vision_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "videomamba" in encoder_name:
            vision_encoder = build_videomamba(self.config.model)
        else:
            raise ValueError(f"not implemented: {encoder_name}")
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name
        logger.info(f"Build text_encoder {encoder_name}")

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder


def load_backbone(args, **kwargs):

    # ----------------- creating config -----------------
    def parse(d):
        x = SimpleNamespace()
        _ = [setattr(x, k,
                     parse(v) if isinstance(v, dict)
                     else [parse(e) for e in v] if isinstance(v, list)
                     else v) for k, v in d.items()]
        return x
    TextEncoders = dict()
    TextEncoders["bert"] = dict(
        name="bert_base",
        pretrained="bert-base-uncased",
        config=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbones/bert/config_bert.json"),
        d_model=768,
        fusion_layer=9,
    )
    TextEncoders["bert_large"] = dict(
        name="bert_large",
        pretrained="bert-large-uncased",
        config=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbones/bert/config_bert_large.json"),
        d_model=1024,
        fusion_layer=19,
    )
    config = {
        'gradient_checkpointing': True,
        'model': dict(
            model_cls="UMT_VIDEOMAMBA",
            vision_encoder=dict(
                # backbone
                name="videomamba_middle",
                img_size=224,
                patch_size=16,
                depth=32,
                embed_dim=576,
                drop_path_rate=0.25,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                fused_add_norm=True,
                rms_norm=True,
                residual_in_fp32=True,
                bimamba_type="v2",
                pool_type="cls+avg",
                kernel_size=1,
                num_frames=args.num_frames,
                ckpt_num_frame=8,
                use_checkpoint=False,
                checkpoint_num=0,
                clip_decoder_embed_dim=576,
                clip_output_dim=512,
                clip_norm_type='l2',
                clip_return_layer=1,
                clip_student_return_interval=1,
                pretrained=None,
                # clip teacher
                clip_teacher="none",
                clip_img_size=224,
                clip_return_interval=1,
                # mask
                video_mask_type="none",
                video_mask_ratio=0.,
                video_double_mask_ratio=0.,
                image_mask_type="none",
                image_mask_ratio=0.,
                image_double_mask_ratio=0.,
                # for ret
                keep_temporal=True,
            ),
            text_encoder=TextEncoders["bert"],
            multimodal=dict(enable=True),
            embed_dim=512,
            temp=0.07,
        ),
    }
    config = parse(config)

    # ----------------- loading model -----------------

    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    model = UMT_VIDEOMAMBA(config=config, tokenizer=tokenizer, is_pretrain=False)

    # ----------------- loading checkpoint -----------------

    state_dict = torch.load(args.backbone_checkpoint, map_location='cpu')
    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing_keys", missing_keys)
    print("unexpected_keys", unexpected_keys)
    print(f"Backbone {args.model} is loaded!")

    return model
