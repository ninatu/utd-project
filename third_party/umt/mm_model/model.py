from torch import nn
import os
import os.path as osp
import torch
from collections import OrderedDict
from types import SimpleNamespace

from .backbones.bert.tokenization_bert import BertTokenizer
from .backbones.vit import build_vit
from .backbones.bert.builder import build_bert


class UMT(nn.Module):
    """docstring for UMT"""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super(UMT, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.d_model
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # criterions
        # self.loss_weight = config.criterion.loss_weight
        # self.criterion_uta = UTA_Loss(
        #     config.criterion.uta_norm_type,
        #     config.criterion.uta_loss_type,
        # )
        # self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)
        # self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)

    # def forward(self, image, text, idx):
    #     """forward and calculate loss.
    #
    #     Args:
    #         image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
    #         text (dict): TODO
    #         idx (torch.Tensor): TODO
    #
    #     Returns: TODO
    #
    #     """
    #     self.clip_contrastive_temperature()
    #
    #     vision_embeds, pooled_vision_embeds = self.encode_vision(image)
    #     text_embeds, pooled_text_embeds = self.encode_text(text)
    #
    #     # obtain vision and text representations.
    #     vision_proj = self.vision_proj(pooled_vision_embeds)
    #     text_proj = self.text_proj(pooled_text_embeds)

        # calculate loss

        # ## VTC loss
        # if self.loss_weight.vtc != 0:
        #     loss_vtc = self.criterion_vtc_vtm.vtc_loss(
        #         vision_proj, text_proj, idx, self.temp, all_gather=True
        #     )
        # else:
        #     loss_vtc = torch.tensor(0)
        #
        # ## VTM loss
        # if self.loss_weight.vtm != 0:
        #     loss_vtm = self.criterion_vtc_vtm.vtm_loss(
        #         self.get_text_encoder(),
        #         self.itm_head,
        #         self.temp,
        #         vision_embeds,
        #         text_embeds,
        #         vision_proj,
        #         text_proj,
        #         text.attention_mask,
        #         idx,
        #     )
        # else:
        #     loss_vtm = torch.tensor(0)
        #
        # return dict(
        #     loss_vtc=loss_vtc * self.loss_weight.vtc,
        #     loss_vtm=loss_vtm * self.loss_weight.vtm,
        # )

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
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config.model)
        else:
            raise ValueError(f"not implemented: {encoder_name}")

        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name

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


def setup_model(config, pretrain=False, find_unused_parameters=False):
    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    model = UMT(config=config, tokenizer=tokenizer, is_pretrain=pretrain)

    if osp.isfile(config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        if 'model' in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for key in state_dict:
            if key.startswith('text_encoder.bert.'):
                new_state_dict[key.replace('text_encoder.bert.', 'text_encoder.')] = state_dict[key]
            elif key.startswith('clip_teacher.'):
                pass
            elif key.startswith('vision_encoder.clip_decoder.'):
                pass
            else:
                new_state_dict[key] = state_dict[key]
        state_dict = new_state_dict

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        print("No pretrained checkpoint provided, training from scratch")

    return model


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
    if args.model == 'vit_b16':
        d_model = 768
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        drop_path_rate = 0.2
        checkpoint_num = 12
        clip_decoder_embed_dim = 768
        clip_output_dim = 512
        clip_img_size=224
        embed_dim = 512
        text_encoder=TextEncoders['bert']
    elif args.model == 'vit_l14':
        d_model = 1024
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        drop_path_rate = 0.3
        checkpoint_num = 24
        clip_decoder_embed_dim = 1024
        clip_output_dim = 768
        clip_img_size = 196
        embed_dim = 768
        text_encoder = TextEncoders['bert_large']
    else:
        raise NotImplementedError


    config = {
        'gradient_checkpointing': True,
        'model': dict(
            model_cls="UMT",
            vision_encoder=dict(
                # backbone
                name=args.model,
                img_size=224,
                patch_size=16,
                d_model=d_model,
                encoder_embed_dim=encoder_embed_dim,
                encoder_depth=encoder_depth,
                encoder_num_heads=encoder_num_heads,
                drop_path_rate=drop_path_rate,
                num_frames=args.num_frames,
                tubelet_size=1,
                use_checkpoint=True,
                checkpoint_num=checkpoint_num,
                clip_decoder_embed_dim=clip_decoder_embed_dim,
                clip_output_dim=clip_output_dim,
                clip_return_layer=0,
                clip_student_return_interval=1,
                pretrained=None,
                # clip teacher
                clip_teacher="none",
                clip_img_size=clip_img_size,
                clip_return_interval=1,
                # mask
                video_mask_type="attention",
                video_mask_ratio=0.,
                video_double_mask_ratio=0.,
                image_mask_type="attention",
                image_mask_ratio=0.,
                image_double_mask_ratio=0.,
                # for ret
                keep_temporal=True,
            ),
            text_encoder=text_encoder,
            multimodal=dict(enable=True),
            embed_dim=embed_dim,
            temp=0.07,
        ),
    }
    config = parse(config)

    # ----------------- loading model -----------------

    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    model = UMT(config=config, tokenizer=tokenizer, is_pretrain=False)

    # ----------------- loading checkpoint -----------------

    checkpoint_model = torch.load(args.backbone_checkpoint, map_location='cpu')
    if 'model' in checkpoint_model.keys():
        state_dict = checkpoint_model["model"]
    else:
        state_dict = checkpoint_model

    new_state_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('text_encoder.bert.'):
            new_state_dict[key.replace('text_encoder.bert.', 'text_encoder.')] = state_dict[key]
        elif key.startswith('clip_teacher.'):
            pass
        elif key.startswith('vision_encoder.clip_decoder.'):
            pass
        else:
            new_state_dict[key] = state_dict[key]
    state_dict = new_state_dict

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    print(f"Model {args.model} is loaded!")

    return model
