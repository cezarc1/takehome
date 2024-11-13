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
        """Load labeled golden chat histories from JSON file."""
        with open(file_path, 'r') as f:
            conversations = json.load(f)
        return [
            cls(messages=[
                ChatMessage(**msg) for msg in conv['chat_history']['messages']
            ]) for conv in conversations
        ]

    def __str__(self):
        messages = []
        for message in self.messages:
            messages.append(str(message))
        return "\n".join(messages)


class LabeledChatHistory(BaseModel):
    """A chat history with a labeled output response."""
    chat_history: ChatHistory
    output: str

    @classmethod
    def load_labeled_histories(
        cls,
        file_path: str = 'training_data/conversations.json'
    ) -> List['LabeledChatHistory']:
        """Load conversation examples from JSON file."""
        with open(file_path, 'r') as f:
            conversations = json.load(f)
        return [
            cls(chat_history=ChatHistory(messages=[
                ChatMessage(**msg) for msg in conv['chat_history']['messages']
            ]),
                output=conv['output']) for conv in conversations
        ]

    def to_dspy_example(self) -> Example:
        """Convert this labeled chat history into a DSPy Example.
        
        The example will contain:
        - input: the chat history up to the last fan message
        - output: the labeled creator response
        """
        example = Example()
        example.input = str(self.chat_history)
        example.output = self.output
        example._input_keys = ["input"]
        example._output_keys = ["output"]
        return example
