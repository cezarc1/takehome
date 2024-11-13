import dspy
from typing import List
from models import ChatHistory, LabeledChatHistory
from .responder import ResponderModule
from .knn_optimizer import KNNOptimizerModule


class ChatterModule(dspy.Module):

    def __init__(self, examples: List[LabeledChatHistory]):
        super().__init__()

        optimizer = KNNOptimizerModule(examples)
        self.responder = optimizer.compile_module(ResponderModule())

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.responder(chat_history=chat_history)
