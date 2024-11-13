import dspy
from typing import List, Optional

import os
from models import ChatHistory
from .responder import ResponderModule
from dspy.teleprompt import KNNFewShot
from dspy.predict import KNN
from dspy import Signature, InputField, OutputField


class ChatterModule(dspy.Module):

    def __init__(self, examples: List[ChatHistory]):
        super().__init__()

        training_examples = [
            example for chat_history in examples
            for example in chat_history.to_dspy_examples()
        ]
        self.optimizer = KNNFewShot(k=3,
                                    trainset=training_examples,
                                    metric="similarity")
        self.compiled_responder = self.optimizer.compile(ResponderModule())

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.compiled_responder(chat_history=chat_history)
