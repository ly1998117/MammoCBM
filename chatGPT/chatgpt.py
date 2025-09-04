# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import time
from openai import OpenAI


class ChatGPT:
    def __init__(self, api_base, api_key, model,
                 prompts=None,
                 conversation_track=False,
                 temperature=0,
                 top_p=1,
                 stream=False
                 ):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            default_headers={"x-foo": "true"},
        )

        self.conversation_track = conversation_track
        self.conversations = {}
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream

    def __call__(self, user_message=None, prompts=None, user_id=None):
        """
        Make remember all the conversation
        :param old_model: Open AI model
        :param user_id: telegram user id
        :param user_message: text message
        :return: str
        """
        if user_message is not None:
            if isinstance(user_message, str):
                user_message = [{"role": "user", "content": user_message}]

        if not self.conversation_track:
            # Generate response

            return self.generate_response_chatgpt(user_message, prompts=prompts)

        conversation_history, gpt_responses = [], []
        # Get the last 10 conversations and responses for this user
        user_conversations = self.conversations.get(user_id, {'conversations': [], 'responses': []})
        user_messages = user_conversations['conversations'] + [user_message]
        gpt_responses = user_conversations['responses']

        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}

        # Construct the full conversation history in the user:assistant, " format
        for i in range(min(len(user_messages), len(gpt_responses))):
            conversation_history.append(user_messages[i])
            conversation_history.append(gpt_responses[i])

        # Add last prompt
        conversation_history.append(user_message)

        # Generate response
        response = self.generate_response_chatgpt(conversation_history, prompts=prompts)

        # Add the response to the user's responses
        gpt_responses.append(response)
        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}
        return response

    def generate_response_chatgpt(self, message_list, prompts=None, times=2):
        if prompts is None:
            prompts = self.prompts
        if prompts is None:
            prompts = []
        if message_list is None:
            message_list = []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompts + message_list,
                stream=self.stream,
                temperature=self.temperature,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=self.top_p
            )
        except Exception as e:
            time.sleep(times)
            print(e)
            return self.generate_response_chatgpt(message_list, prompts, times + 1)
        if self.stream:
            return response
        return response.choices[0].message.content.strip()
