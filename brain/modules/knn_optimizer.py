from typing import List

from dspy import Module
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory


class KNNOptimizerModule(Module):

    def __init__(self, examples: List[LabeledChatHistory], k: int = 3):
        super().__init__()
        self.training_examples = [
            example.to_dspy_example() for example in examples
        ]
        # TODO: Add metric and split for test sets/
        self.optimizer = KNNFewShot(k, trainset=self.training_examples)

    def compile_module(self, module: Module) -> Module:
        return self.optimizer.compile(module, trainset=self.training_examples)
