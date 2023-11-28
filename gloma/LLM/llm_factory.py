from .llama import LLAMA
from .chat_gpt import ChatGPT


class LLMFactory:
    @staticmethod
    def create_chat_object(chat_type, use_cuda=False):
        if chat_type.lower() == 'chatgpt':
            return ChatGPT()
        elif chat_type.lower() == 'llama':
            return LLAMA(cuda=use_cuda)
        else:
            raise ValueError(f"Unknown chat type: {chat_type}")
