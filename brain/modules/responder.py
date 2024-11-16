from typing import Optional

from dspy import Module, OutputField, TypedChainOfThought
from models import ChatHistory
from signatures.responder import Responder


class ResponderModule(Module):

    def __init__(self):
        super().__init__()
        reasoning = OutputField(
            prefix=
            "Reasoning: Let's think step by step to decide on your message. "
            "Use the previous chat history to help guide your response and "
            "stay within the same voice, tone, language, length and especially "
            "style as previous chat histories, if provided. Only use the 1st "
            "person and never include any other details, including your "
            "thoughts, reasoning, etc when writing your response back to the "
            "fan.",
            desc="Reasoning for the response.",
        )
        self.prog = TypedChainOfThought(Responder, reasoning=reasoning)

    def forward(
        self,
        chat_history: ChatHistory,
        img_base64: Optional[str] = None,
    ):
        return self.prog(chat_history=chat_history, img_base64=img_base64)
