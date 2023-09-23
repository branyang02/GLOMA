import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_13b'

class LLAMA:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
        )

    def query_message(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=32)
        return self.tokenizer.decode(generation_output[0])
