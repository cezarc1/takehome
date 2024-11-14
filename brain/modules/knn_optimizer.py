from typing import List

from dspy import Module
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory


class KNNOptimizerModule(Module):

    def __init__(self, examples: list[LabeledChatHistory], k: int = 3):
        super().__init__()
        self.training_examples = [
            example.to_dspy_example() for example in examples
        ]
        # TODO: Add metric and split for test sets/
        self.optimizer = KNNFewShot(
            k,
            # for some reason KNNFewShot uses dsp.Example vs dspy.Example :shrug:
            trainset=self.training_examples  # type: ignore
        )

    def compile_module(self, module: Module) -> Module:
        return self.optimizer.compile(module, trainset=self.training_examples)
