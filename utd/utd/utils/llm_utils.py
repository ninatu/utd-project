import torch


def construct_final_prompt(model_id, tokenizer, system_prompt, prompt, answer_prompt):
    if 'mistralai' in model_id:
        main_prompt = f'{system_prompt.strip()} {prompt.strip()}'.strip()
        answer_prompt = answer_prompt.strip()
        final_prompt = f"""<s>[INST] {main_prompt} [/INST] {answer_prompt}""".strip()
    elif 'NousResearch' in model_id:
        messages = [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": prompt.strip()
            }
        ]

        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        final_prompt = final_prompt + f'<|im_start|>assistant {answer_prompt}'
    else:
        raise NotImplementedError

    return final_prompt


def contract_prompt_with_examples(model_id, tokenizer, system_prompt, main_prompt_template, text_prompt, answer_prompt, EXAMPLES,
                                  input_text):
    final_prompt = ""
    for example, answer in EXAMPLES:
        example = text_prompt.format(*example) if isinstance(example, list) else text_prompt.format(example)
        main_prompt = main_prompt_template.format(example)
        cur_prompt = construct_final_prompt(model_id, tokenizer, system_prompt, main_prompt, answer_prompt)
        final_prompt += cur_prompt + "\n" + answer.strip() + tokenizer.eos_token
    input_text = text_prompt.format(*input_text) if isinstance(input_text, list) else text_prompt.format(input_text)
    main_prompt = main_prompt_template.format(input_text)
    cur_prompt = construct_final_prompt(model_id, tokenizer, system_prompt, main_prompt, answer_prompt)
    return final_prompt + cur_prompt


def generate_greedly(model, tokenizer, input_ids, past_key_values, max_new_tokens):
    generated = []
    input_ids = input_ids.to(model.device)

    for i in range(max_new_tokens):
        with torch.inference_mode():
            output = model(input_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values

        token = torch.argmax(output.logits[..., -1, :])
        if token == tokenizer.eos_token_id:
            break
        input_ids = token.unsqueeze(0).unsqueeze(0)
        generated += [token.tolist()]
    return generated
