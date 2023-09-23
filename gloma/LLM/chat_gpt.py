import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

class ChatGPT:
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are an intelligent assistant."}]

    def query_message(self, message):
        if message:
            self.messages.append({"role": "user", "content": message})
            chat = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo", messages=self.messages
                model="gpt-4", messages=self.messages
            )
            reply = chat.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            return reply
