import numpy as np
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import bfloat16
import tqdm
import os
import pickle

from utd.utd.utils.llm_utils import contract_prompt_with_examples, generate_greedly
from utd.utd.utils.utils import load_descriptions
from utd.utd.utils.extract_concepts_utils import get_prompts_and_examples, parse_llm_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_description_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--save_raw_llm_output_path", type=str, default=None)

    parser.add_argument("--num_data_chunks", type=int, default=None, help="Total number of parts to split the dataset into")
    parser.add_argument("--chunk_id", type=int, default=None, help="Index of the data split to process (0-based)")

    parser.add_argument("--concept", type=str, choices=['objects', 'activities', 'verbs','objects+composition+activities_15_words'])

    parser.add_argument("--model_id", type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument("--token", default=None, help='hugging_face token')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_cache", action='store_true', default=False)

    args = parser.parse_args()

    input_description_path = args.input_description_path
    output_path = args.output_path
    save_raw_llm_output_path = args.save_raw_llm_output_path
    concept = args.concept

    chunk_id = args.chunk_id
    num_data_chunks = args.num_data_chunks
    model_id = args.model_id

    dataset = load_descriptions(input_description_path)
    dataset = list(sorted(dataset.items()))
    if chunk_id is not None:
        step = int(np.ceil(len(dataset) / num_data_chunks))
        dataset = dataset[step * chunk_id: step * (chunk_id + 1)]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=bfloat16,
                                                 device_map='auto',
                                                 token=args.token)

    system_prompt, main_prompt_template, text_prompt, answer_prompt, EXAMPLES, max_new_tokens = \
        get_prompts_and_examples(concept)

    COMMON_PAST_KEY_VALUES = None
    outputs = {}
    raw_outputs = {}

    for video_id, texts in tqdm.tqdm(dataset):
        cur_outputs = []
        cur_raw_outputs = []

        texts = texts['objects+composition+activities']

        for text in texts:
            tmp = "INPUTTEXT"
            tmp_input = [tmp] * len(text) if isinstance(text, list) else tmp
            prompt = contract_prompt_with_examples(model_id, tokenizer, system_prompt, main_prompt_template, text_prompt, answer_prompt,
                                                   EXAMPLES, tmp_input)
            prompt_parts = prompt.split(tmp)
            assert len(prompt_parts) == (len(tmp_input) + 1 if isinstance(tmp_input, list) else 2)
            COMMON_PROMPT = prompt_parts[0]
            cur_prompt = prompt_parts[-1]
            text = text_prompt.format(*text) if isinstance(text, list) else text_prompt.format(text)
            cur_prompt = text + cur_prompt
            if args.debug:
                print('--------------------------------- PROMPT ---------------------------------')
                print(COMMON_PROMPT + cur_prompt)

            if args.no_cache:
                input_ids = tokenizer(COMMON_PROMPT + cur_prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
                output = generate_greedly(model, tokenizer, input_ids, None, max_new_tokens)
            else:
                # we cache common part of prompt to speed up inference, but this is not working correctly with any version of libs
                if COMMON_PAST_KEY_VALUES is None:
                    COMMON_INPUT_IDS = tokenizer(COMMON_PROMPT, return_tensors="pt")['input_ids']
                    with torch.inference_mode():
                        COMMON_PAST_KEY_VALUES = model(COMMON_INPUT_IDS.to(model.device),
                                                       use_cache=True)['past_key_values']

                input_ids = tokenizer(cur_prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
                output = generate_greedly(model, tokenizer, input_ids, COMMON_PAST_KEY_VALUES, max_new_tokens)
            output = tokenizer.decode(output, skip_special_tokens=True)
            cur_raw_outputs.append(output)
            if args.debug:
                print('--------------------------------- LLM OUTPUT ---------------------------------')
                print(output)
            output = parse_llm_output(output.strip(), concept)
            cur_outputs.append(output)
            if args.debug:
                print('--------------------------------- PARSED OUTPUT ---------------------------------')
                print(output)
        outputs[video_id] = {concept: cur_outputs}
        raw_outputs[video_id] = {concept: cur_raw_outputs}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as fout:
        pickle.dump(outputs, fout)

    if save_raw_llm_output_path is not None:
        os.makedirs(os.path.dirname(save_raw_llm_output_path), exist_ok=True)
        with open(save_raw_llm_output_path, 'wb') as fout:
            pickle.dump(outputs, fout)
