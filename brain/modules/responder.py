from dspy import Module, OutputField, TypedChainOfThought
from models import ChatHistory
from signatures.responder import Responder


class ResponderModule(Module):

    def __init__(self):
        super().__init__()
        reasoning = OutputField(
            prefix=
            "Reasoning: Let's think step by step to decide on our message.", )
        self.prog = TypedChainOfThought(Responder, reasoning=reasoning)

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.prog(chat_history=chat_history)
