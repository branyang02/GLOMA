import torch
import os
from transformers import LlamaTokenizer, LlamaForCausalLM
from utils.helper import extract_json_content
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
login(os.getenv('HF_API'))

model_id="meta-llama/Llama-2-13b-chat-hf"
custom_cache_directory = os.getenv('CACHE_DIR')
peft_model_id = os.getenv('PEFT_MODEL')


class LLAMA:
    def __init__(self):
    
        self.tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir=custom_cache_directory)
        self.model = LlamaForCausalLM.from_pretrained(
            model_id, 
            load_in_8bit=True, 
            device_map='auto', 
            torch_dtype=torch.float16, 
            cache_dir=custom_cache_directory
        )
        self.model.load_adapter(peft_model_id)
        print("🦙🦙🦙 LLAMA Initialized! 🦙🦙🦙")
        
    def query_message(self, prompt):
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = model_input.input_ids.size(1)  # Get the length of the input sequence

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=100)[0]
            generated_sequence = output[input_length:]  # Extract only the generated tokens

            return_message = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            json_content = extract_json_content(return_message)
            return json_content