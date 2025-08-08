import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from . import heads
from . import base_vision_transformer as vit
import torch.nn.functional as F
import math


class VCOPHeader(torch.nn.Module):
    def __init__(self, tuple_len=3, feature_size=768):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VCOPHeader, self).__init__()
        self.feature_size = feature_size
        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        self.tuple_len = tuple_len
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.class_num = math.factorial(tuple_len)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        """
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i + 1, self.tuple_len):
                pf.append(torch.cat([x[:, i], x[:, j]], dim=1))
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits
        return h


class AllinoneTransformerSS(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        # self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        # self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, self.config["vit"])(
            pretrained=False, config=self.config
        )
        self.embed_dim = config["hidden_size"]

        self.pooler = heads.Pooler(config["hidden_size"])
        # self.pooler.apply(objectives.init_weights)

        # num frames
        self.num_frames = config["num_frames"]  # a global variable to identify if image/video

        # self.temporal_roll_module = TemporalRoll(n_segment=self.num_frames, v=0)

    def get_num_layers(self):
        return len(self.transformer.blocks)

    def infer(
        self,
        x,
    ):
        img = x
        img = img.contiguous().view(-1, img.size()[2], img.size()[3], img.size()[4])  # btchw to [bt]chw
        (image_embeds, image_masks, patch_index, image_labels) = self.transformer.visual_embed(
            img, max_image_len=self.config["max_image_len"], mask_it=False)

        image_token_type_idx = 1
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))

        x = image_embeds
        co_masks = image_masks

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
        x = self.transformer.norm(x)
        x = x.view(x.size(0) // self.num_frames, -1, x.size(-2),
                                     x.size(-1))
        x = x.view(x.size(0), -1, x.size(-1))
        return x

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.infer(x)

    def _inflate_positional_embeds(self, new_state_dict, load_temporal_fix='zeros'):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'transformer.temporal_embed' in new_state_dict and 'transformer.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['transformer.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.config['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.config["vit"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.config["vit"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {load_temporal_fix}')
                    if load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['transformer.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'transformer.pos_embed' in new_state_dict and 'transformer.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['transformer.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['transformer.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


def load_backbone(args, **kwargs):
    config = {'image_size': 224,
              'patch_size': 16,
              'num_frames': 8,
              'max_text_len': 40,
              'vocab_size': 30522,
              'vit': 'vit_base_patch16_224',
              'hidden_size': 768,
              'num_heads': 12,
              'num_layers': 12,
              'mlp_ratio': 4,
              'drop_rate': 0.1,
              'shared_embedding_dim': 512,
              'max_image_len': -1}

    model = AllinoneTransformerSS(config)

    checkpoint_model = torch.load(args.backbone_checkpoint, map_location='cpu')
    checkpoint_model = model._inflate_positional_embeds(checkpoint_model)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)

    assert set(missing_keys).difference(['transformer.temporal_embed']) == set()
    print(unexpected_keys)
    assert set(unexpected_keys).difference([
        'mlm_score.bias', 'mlm_score.transform.dense.weight', 'mlm_score.transform.dense.bias',
        'mlm_score.transform.LayerNorm.weight', 'mlm_score.transform.LayerNorm.bias',
        'mlm_score.decoder.weight', 'vtm_score.fc.weight', 'vtm_score.fc.bias', 'itm_score.fc.weight',
        'itm_score.fc.bias', 'text_embeddings.position_ids']) == set()
    print(f"Backbone {args.model} is loaded!")
    return model