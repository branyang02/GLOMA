import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import torch

model_path = os.getenv('ADAPTER_MODEL_PATH')

class LLAMA:
    def __init__(self, cuda=True):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()
        
    def query_message(self, prompt):

        if "transformation:" not in prompt:
            prompt = prompt.replace("Q:", "### INSTRUCTION: ")
            prompt = prompt.replace("sentence:", "sentence: ### INPUT: ")
            prompt = prompt.replace("A:", "### RESPONSE: ")
        else:
            prompt = prompt.replace("Q:", "### INSTRUCTION: ")
            prompt = prompt.replace("transformation:", "transformation: \n### INPUT: ")
            prompt = prompt.replace("Output:", "### RESPONSE: ")
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.cuda:
            input_ids = input_ids.to('cuda')
        
        output_tokens = self.model.generate(input_ids, max_new_tokens=75)
        raw_output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # print("RAW MODEL OUTPUT:", raw_output)
        output = self._extract_json_response(raw_output)
        # print("MODEL OUTPUT:", output)

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