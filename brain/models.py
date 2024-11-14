import json
import logging
from datetime import datetime
from typing import List, Optional

from dspy import Example
from pydantic import BaseModel


class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    image_base64: Optional[str] = None  # base64 encoded image
    timestamp: Optional[datetime] = None

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
        previous_message_timestamp: Optional[datetime] = None
        for message in self.messages:
            message_str = str(message)
            # if i == len(self.messages) - 1 and not message.from_creator:
            #     message_str = (
            #         "(The fan just sent the following message which your message must respond to): "
            #         + message_str
            #     )
            time_gap = self.get_time_gap_message(previous_message_timestamp,
                                                 message)
            if time_gap:
                messages.append(time_gap + ":" + message_str)
            else:
                messages.append(message_str)
            previous_message_timestamp = message.timestamp
        return "\n".join(messages)

    def get_time_gap_message(self,
                             previous_message_timestamp: Optional[datetime],
                             current_message: ChatMessage) -> str:
        if not previous_message_timestamp or not current_message.timestamp:
            return ""

        time_gap = self._format_time_gap(previous_message_timestamp,
                                         current_message.timestamp)
        if current_message.from_creator:
            message = f"sent by YOU {time_gap}"
        else:
            message = f"sent by FAN {time_gap}"
        return message

    def _format_time_gap(self, prev_time: datetime,
                         current_time: datetime) -> str:
        """Format the time gap between messages in a human-readable way.
        
        Args:
            prev_time: The timestamp of the previous message
            
        Returns:
            A formatted string describing the time gap, or empty string if gap is small
        """

        time_gap = current_time - prev_time

        if time_gap.days > 0:
            return f"[{time_gap.days} days later]\n"
        elif time_gap.seconds > 3600:
            hours = time_gap.seconds // 3600
            return f"{hours} hours later\n"
        elif time_gap.seconds > 300:  # 5 minutes
            minutes = time_gap.seconds // 60
            return f"{minutes} minutes later\n"
        elif time_gap.seconds > 10:  # 10 seconds
            return f"{time_gap.seconds} seconds later\n"
        return "a few seconds later"

    def model_dump_json(self, **kwargs):
        return str(self)


class LabeledChatHistory(BaseModel):
    """A chat history with a labeled output response."""
    chat_history: ChatHistory
    response: str

    @classmethod
    def load_labeled_histories(
        cls,
        file_path: str = 'training_data/conversations.json'
    ) -> List['LabeledChatHistory']:
        """Load conversation examples from JSON file."""
        logging.info(f"Loading labeled chat histories from {file_path}")
        try:
            with open(file_path, 'r') as f:
                conversations = json.load(f)
        except FileNotFoundError:
            logging.error(f"File {file_path} not found")
            raise
        return [
            cls(chat_history=ChatHistory(messages=[
                ChatMessage(**msg) for msg in conv['chat_history']['messages']
            ]),
                response=conv['output']) for conv in conversations
        ]

    def to_dspy_example(self) -> Example:
        """Convert this labeled chat history into a DSPy Example.
        
        The example will contain:
        - input: the chat history as a structured object
        - output: the labeled creator response
        """
        return Example(
            chat_history=self.chat_history,
            output=self.response,
        ).with_inputs("chat_history")
