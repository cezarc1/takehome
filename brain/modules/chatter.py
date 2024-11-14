from dspy import Module, Prediction
from models import ChatHistory, LabeledChatHistory

from .content_filter import ContentFilterModule
from .knn_optimizer import KNNOptimizer
from .responder import ResponderModule


class ChatterModule(Module):

    def __init__(self, examples: list[LabeledChatHistory]):
        super().__init__()
        optimizer = KNNOptimizer(examples)
        self.responder = optimizer.compile_module(ResponderModule())
        self.content_filter = ContentFilterModule()

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        initial_response = self.responder(chat_history=chat_history)
        filtered = self.content_filter(message=initial_response.response)
        if not filtered.is_safe:
            return Prediction(response=filtered.filtered_message)
        return initial_response
