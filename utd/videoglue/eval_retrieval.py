import os

import torch
import tqdm
import torch.nn.functional as F
import argparse
import itertools
import pandas as pd
import numpy as np
import  json

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from utd.utd.datasets import BaseDataset
from utd.videoglue.metrics import compute_retrieval_metrics_per_query


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--backbone_name', required=True)

    parser.add_argument('--backbone', required=True)
    parser.add_argument('--backbone_checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--no_itm', action='store_true', default=False)

    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset_metadata_root', default=None)
    parser.add_argument('--encoder_batch_size', type=int, default=64)
    parser.add_argument('--fusion_batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--k_test', default=128)
    parser.add_argument('--output_root', default='output/retrieval/')
    parser.add_argument('--debiased_splits_path', type=str, default=None)

    args = parser.parse_args()
    device = args.device

    os.makedirs(args.output_root, exist_ok=True)
    output_root = f'{args.output_root}/{args.dataset}_{args.backbone_name}'
    print("Output root", output_root)
    os.makedirs(output_root, exist_ok=True)

    transform = transforms.Compose([
                transforms.Lambda(lambd=lambda x: torch.tensor(x).permute(0, 3, 1, 2).float() / 255.),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])
    dataset = BaseDataset(root=args.dataset_root, split=args.split, num_frames=args.num_frames, metadata_root=args.dataset_metadata_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.encoder_batch_size, num_workers=1, shuffle=False, drop_last=False,
                            pin_memory=True)

    if args.backbone == 'umt':
        from umt.mm_model.model import load_backbone
        model = load_backbone(args)
        tokenizer = model.tokenizer
    elif args.backbone == 'internvid':
        from internvid.viclip.viclip import load_backbone
        model = load_backbone(args)
        tokenizer = model.tokenizer
    elif args.backbone == 'videomamba':
        from video_mamba.mm_model.umt_videomamba import load_backbone
        model = load_backbone(args)
        tokenizer = model.tokenizer
    else:
        raise NotImplementedError()

    model.eval()
    model.to(device)

    # ------------------------------- contrastive -------------------------------
    all_text_ids = []
    all_text_contrastive_embeds = []
    all_text_feat = []
    all_text_attn = []

    all_video_ids = []
    all_video_contrastive_embeds = []
    all_video_feat = []

    for data in tqdm.tqdm(dataloader):
        if isinstance(data['text'][0], list) or isinstance(data['text'][0], tuple):
            data['text'] = list(itertools.chain.from_iterable(data['text']))
            data['text_video_id'] = list(itertools.chain.from_iterable(data['text_video_id']))
            data['tv_pair_id'] = list(itertools.chain.from_iterable(data['tv_pair_id']))
        text = data['text']
        video = data['video']
        text_video_id = data['text_video_id']
        video_id = data['video_id']

        with torch.no_grad():
            all_text_ids.extend(text_video_id)
            if tokenizer is not None:
                text_input = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
                all_text_attn.append(text_input.attention_mask.cpu())
            else:
                text_input = text
                all_text_attn.append(None)

            text_feat, text_embed = model.get_text_embedding_and_features(text_input)
            text_embed = F.normalize(text_embed, dim=-1).cpu()

            all_text_contrastive_embeds.append(text_embed)
            all_text_feat.append(None if (text_feat is None) else text_feat.cpu())
            all_video_ids.extend(video_id)

            video = video.to(device, non_blocking=True)
            video_feat, video_embed = model.get_vision_embedding_and_features(video)
            video_embed = F.normalize(video_embed, dim=-1).cpu()

            all_video_contrastive_embeds.append(video_embed)
            all_video_feat.append(None if (video_feat is None) else video_feat.cpu())

    all_text_contrastive_embeds = torch.cat(all_text_contrastive_embeds, dim=0)
    all_video_contrastive_embeds = torch.cat(all_video_contrastive_embeds, dim=0)
    sims_matrix = all_text_contrastive_embeds @ all_video_contrastive_embeds.T

    output = compute_retrieval_metrics_per_query(sims_matrix, all_text_ids, all_video_ids)
    output['video_id'] = all_text_ids
    output = pd.DataFrame(output)
    output.to_csv(f'{output_root}/contrastive.csv', index=False)

    def print_metrics(output):
        r1, r5, r10, mr = output['R1'].mean(), output['R5'].mean(), output['R10'].mean(), np.median(output['rank']) + 1
        print('R@1: {:.3f} - R@5: {:.3f} - R@10: {:.3f} - Median R: {}'.format(r1, r5, r10, mr))
        print('{:.3f} {:.3f} {:.3f}'.format(r1, r5, r10))

    print('Contrastive:')
    print_metrics(output)

    if args.debiased_splits_path is not None:
        with open(args.debiased_splits_path) as f:
            debiased_video_ids = json.load(f)['debiased']
        print('Contrastive (debiased split):')
        print_metrics(output[output['video_id'].isin(debiased_video_ids)])

    # ------------------------------- image-text matching -------------------------------

    if not args.no_itm:
        all_text_feat = torch.cat(all_text_feat, dim=0)
        all_text_attn = torch.cat(all_text_attn, dim=0)
        all_video_feat = torch.cat(all_video_feat, dim=0)

        k_test = args.k_test
        batch_size = args.fusion_batch_size
        rank = 0
        num_tasks = 1

        with torch.no_grad():
            score_matrix_t2v = torch.full((len(all_text_ids), len(all_video_ids)), -100.0)

            step = sims_matrix.size(0) // num_tasks + 1
            start = rank * step
            end = min(sims_matrix.size(0), start + step)

            # text --> image
            for i, sims in enumerate(tqdm.tqdm(sims_matrix[start:end])):
                topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
                encoder_output = all_video_feat[topk_idx].to(device, non_blocking=True)

                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)

                encoder_embeds = all_text_feat[[start + i]].repeat(k_test, 1, 1).to(device, non_blocking=True)
                attention_mask = all_text_attn[[start + i]].repeat(k_test, 1).to(device, non_blocking=True)
                encoder_hidden_states = encoder_output.to(device, non_blocking=True)
                encoder_attention_mask = encoder_att.to(device, non_blocking=True)

                for offset in range(0, k_test, batch_size):
                    score = model.get_itm_score(encoder_embeds=encoder_embeds[offset: offset + batch_size],
                                                attention_mask=attention_mask[offset: offset + batch_size],
                                                encoder_hidden_states=encoder_hidden_states[offset: offset + batch_size],
                                                encoder_attention_mask=encoder_attention_mask[offset: offset + batch_size],
                                                )
                    score_matrix_t2v[start + i, topk_idx[offset: offset + batch_size]] = score.cpu()

        output = compute_retrieval_metrics_per_query(score_matrix_t2v, all_text_ids, all_video_ids)
        output['video_id'] = all_text_ids
        output = pd.DataFrame(output)
        output.to_csv(f'{output_root}/reranked.csv', index=False)

        print('Reranked:')
        print_metrics(output)

        if args.debiased_splits_path is not None:
            with open(args.debiased_splits_path) as f:
                debiased_video_ids = json.load(f)['debiased']
            print('Reranked (debiased split):')
            print_metrics(output[output['video_id'].isin(debiased_video_ids)])


