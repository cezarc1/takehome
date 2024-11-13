from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
import json
from dsp.primitives import Example
from dspy import Signature, InputField, OutputField


class ChatMessage(BaseModel):
    from_creator: bool
    content: str

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        message = role + ": " + self.content
        return message


class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    @classmethod
    def load_chat_histories(
        cls,
        file_path: str = 'training_data/conversations.json'
    ) -> List['ChatHistory']:
        """Load conversation examples from JSON file and return list of ChatHistory objects."""
        with open(file_path, 'r') as f:
            conversations = json.load(f)
        return [
            cls(messages=[
                ChatMessage(**msg) for msg in conv['chat_history']['messages']
            ]) for conv in conversations
        ]

    def to_dspy_examples(self) -> List[Example]:
        """Convert this chat history into a list of DSPy Examples.
        
        Creates examples where:
        - input is a fan's message
        - output is the matching creator's response that follows
        Each example includes the required _input_keys and _output_keys for DSPy KNN
        """
        examples = []
        for i in range(0,
                       len(self.messages) - 1,
                       2):  # Step by 2 since we know fan->creator pattern
            fan_msg = self.messages[i]
            creator_msg = self.messages[i + 1]

            example = Example(input=fan_msg.content,
                              output=creator_msg.content,
                              _input_keys=["input"],
                              _output_keys=["output"])
            examples.append(example)

        return examples

    def __str__(self):
        messages = []
        for i, message in enumerate(self.messages):
            message_str = str(message)
            # if i == len(self.messages) - 1 and not message.from_creator:
            #     message_str = (
            #         "(The fan just sent the following message which your message must respond to): "
            #         + message_str
            #     )
            messages.append(message_str)
        return "\n".join(messages)

    def model_dump_json(self, **kwargs):
        return str(self)
