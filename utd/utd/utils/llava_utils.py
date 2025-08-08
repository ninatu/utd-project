import torch
from typing import List
from PIL import Image
import re

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IMAGE_PLACEHOLDER
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images, KeywordsStoppingCriteria
from llava.conversation import SeparatorStyle, conv_templates


# Based on: https://github.com/haotian-liu/LLaVA/issues/407
class LLaVaProcessor:
    def __init__(self, tokenizer, image_processor, model_name, model_config):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode

    def _format_text(self, qs, conv=None):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model_config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model_config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if conv is None:
            conv = conv_templates[self.conv_mode].copy()
        else:
            conv = conv.copy()

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt

    def _load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    def _convert_image_to_pil(self, image):
        return Image.fromarray(image).convert("RGB")

    @staticmethod
    def _pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths=None, images=None, conv=None):
        prompt = [self._format_text(text, conv=conv) for text in batch_text]
        if image_paths is not None:
            images = [self._load_image(image_path) for image_path in image_paths]
        else:
            images = [self._convert_image_to_pil(image) for image in images]
        image_sizes = [image.size for image in images]

        images_tensor = process_images(
            images,
            self.image_processor,
            self.model_config
        )

        input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self._pad_sequence_to_max_length(seq, max_len) for seq in input_ids]
        input_ids = torch.stack(padded_input_ids)

        return images_tensor, image_sizes, input_ids


def predict(model, processor, input_ids, image_tensor, image_sizes, eval_params):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.cuda(),
            images=image_tensor.to(model.device, dtype=torch.float16),
            image_sizes=image_sizes,
            use_cache=True,
            **eval_params
        )

    generated_outputs = processor.tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )
    return generated_outputs