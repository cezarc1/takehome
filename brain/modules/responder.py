from dspy import Module, OutputField, TypedChainOfThought
from models import ChatHistory
from signatures.responder import Responder


class ResponderModule(Module):

    def __init__(self):
        super().__init__()
        reasoning = OutputField(
            prefix=
            "Reasoning: Let's think step by step to decide on our message. "
            "Use the previous chat history to help guide our response and "
            "stay within the same voice and tone as previous chat histories "
            "provided, as much as possible. Pay close attention to the "
            "voice and tone of your previous replies where THE FAN asked "
            "similar questions. YOU should answer as closely as possible to "
            "the voice and tone of your previous replies.",
            desc="Reasoning for the response.",
        )
        self.prog = TypedChainOfThought(Responder, reasoning=reasoning)

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.prog(chat_history=chat_history)
