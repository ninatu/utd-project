import tqdm
import os
import pickle
import argparse

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from utd.utd.datasets import BaseDataset, MSRVTTDataset, LSMDCDataset, YouCookDataset, MSVDDataset, DIDEMODataset
from utd.utd.utils.llava_utils import LLaVaProcessor, predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--metadata_root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=['train', 'test', 'val', 'testlist01', 'trainlist01'])
    parser.add_argument("--num_data_chunks", type=int, default=None, help="Total number of parts to split the dataset into")
    parser.add_argument("--chunk_id", type=int, default=None, help="Index of the data split to process (0-based)")
    
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--load_8bit", action='store_true', default=False)

    args = parser.parse_args()
    split = args.split
    num_data_chunks = args.num_data_chunks
    chunk_id = args.chunk_id

    temperature = args.temperature
    batch_size = args.batch_size
    output_path = args.output_path

    if output_path is None:
        output_path = f'output/UTD_descriptions/{args.dataset}_{split}_{args.model_path.replace("liuhaotian/", "")}_nf{args.num_frames}' \
                      f'_t{temperature:.1f}{"_fp"if not args.load_8bit else ""}' \
                      f'{f"_part_{chunk_id}_{num_data_chunks}" if chunk_id is not None else ""}'\
                          .replace('.', '_') + '.pickle'
        print('Outfile:', output_path, flush=True)

    DatasetTypes = {
        'msrvtt': MSRVTTDataset,
        'lsmdc': LSMDCDataset,
        'youcook': YouCookDataset,
        'msvd': MSVDDataset,
        'didemo': DIDEMODataset,
    }

    DatasetType = DatasetTypes.get(args.dataset, BaseDataset)
    dataset = DatasetType(args.dataset_root, split,
                          num_frames=args.num_frames, part=chunk_id, n_parts=num_data_chunks, metadata_root=args.metadata_root, safe_read=True)

    prompt = "Describe the objects relationships in the photo. "
    max_new_tokens = 1000

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        load_8bit=args.load_8bit
    )

    model_name = get_model_name_from_path(args.model_path)
    processor = LLaVaProcessor(tokenizer, image_processor, model_name, model.config)

    eval_params = {
        "max_new_tokens": max_new_tokens
    }

    if temperature > 0:
        eval_params["do_sample"] = True
        eval_params["temperature"] = temperature
    else:
        eval_params["do_sample"] = False

    if batch_size is None:
        batch_size = args.num_frames

    outputs = {}
    for data in tqdm.tqdm(dataset):
        video_id = data['video_id']
        images = data["video"]

        image_tensor, image_sizes, input_ids = processor.get_processed_tokens_batch([prompt], images=images)
        input_ids = input_ids.repeat(len(image_tensor), 1)
        prediction = []

        for i in range(len(image_tensor) // batch_size):
            cur_pred = predict(model, processor,
                               input_ids[i * batch_size: (i + 1) * batch_size],
                               image_tensor[i * batch_size: (i + 1) * batch_size],
                               image_sizes[i * batch_size: (i + 1) * batch_size],
                               eval_params)
            prediction.extend(cur_pred)
        outputs[video_id] = {'objects+composition+activities': prediction}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as fout:
        pickle.dump(outputs, fout)

