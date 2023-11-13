# import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM

# model_path = 'openlm-research/open_llama_13b'

# class LLAMA:
#     def __init__(self):
#         self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
#         self.model = LlamaForCausalLM.from_pretrained(
#             model_path, torch_dtype=torch.float16, device_map='auto',
#         )

#     def query_message(self, prompt):
#         print(prompt)

#         # prompt = prompt.replace("Q:", "### INSTRUCTION: ")
#         # prompt = prompt.replace("A:", "### RESPONSE: ")

#         # prompt = prompt.replace("Now, ", "### INSTRUCTION: ")
#         # prompt = prompt.replace("Output:", "### RESPONSE: ")

#         input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to('cuda')
#         generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=32)
#         return self.tokenizer.decode(generation_output[0])

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json

os.environ['TRANSFORMERS_CACHE'] = '/scratch/rhm4nj/HF_Models'
os.environ['HF_HOME'] = '/scratch/rhm4nj/HF_Models'

model_path = '../checkpoints/GLOMA-llama-2-13b-v1-checkpoint-831'
# model_path = '/scratch/rhm4nj/GLOMA/models/GLOMA-llama-2-13b-v1/checkpoint-554'
# model_path = '/scratch/rhm4nj/GLOMA/models/GLOMA-llama-2-13b-v1/checkpoint-831'

class LLAMA:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.model = self.model.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def query_message(self, prompt):

        if "transformation:" not in prompt:
            prompt = prompt.replace("Q:", "### INSTRUCTION: ")
            prompt = prompt.replace("sentence:", "sentence: ### INPUT: ")
            # prompt = prompt.replace("A:", "### RESPONSE: ")
        else:
            prompt = prompt.replace("Q:", "### INSTRUCTION: ")
            prompt = prompt.replace("transformation:", "transformation: \n### INPUT: ")
            # prompt = prompt.replace("Output:", "### RESPONSE: ")
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # input_ids = input_ids.to('cuda')
        output_tokens = self.model.generate(input_ids, max_new_tokens=75)
        raw_output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print("RAW MODEL OUTPUT:", raw_output)
        output = self._extract_json_response(raw_output)
        print("MODEL OUTPUT:", output)

        return output
    
    def _extract_json_response(self, text):
        pattern = r'### RESPONSE:.*?(\{.*?\})'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            pattern = r'### A:.*?\{(.+?)\}'
            match = re.search(pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
            json_str = json_str.replace("'", "\"")
            return json_str
