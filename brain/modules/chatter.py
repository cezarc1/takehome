import dspy
from typing import List
from models import ChatHistory, LabeledChatHistory
from .responder import ResponderModule
from .knn_optimizer import KNNOptimizerModule
from .content_filter import ContentFilterModule


class ChatterModule(dspy.Module):

    def __init__(self, examples: List[LabeledChatHistory]):
        super().__init__()

        optimizer = KNNOptimizerModule(examples)
        self.responder = optimizer.compile_module(ResponderModule())
        self.content_filter = ContentFilterModule()

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        initial_response = self.responder(chat_history=chat_history)
        filtered = self.content_filter(message=initial_response.output)
        # we only return the filtered message if it's deemed unsafe.
        if not filtered.is_safe:
            return dspy.Prediction(output=filtered.filtered_message)
        return initial_response
