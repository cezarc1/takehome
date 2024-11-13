import dspy

from signatures.responder import Responder
from models import ChatHistory


class ResponderModule(dspy.Module):

    def __init__(self):
        super().__init__()
        reasoning = dspy.OutputField(
            prefix=
            "Reasoning: Let's think step by step to decide on our message.", )
        self.prog = dspy.TypedChainOfThought(Responder, reasoning=reasoning)

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.prog(chat_history=chat_history)
