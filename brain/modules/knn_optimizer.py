import dspy
from typing import List
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory


class KNNOptimizerModule(dspy.Module):

    def __init__(self, examples: List[LabeledChatHistory], k: int = 3):
        super().__init__()
        training_examples = [example.to_dspy_example() for example in examples]
        # TODO: Add metric and split for test sets/
        self.optimizer = KNNFewShot(k, trainset=training_examples)

    def compile_module(self, module: dspy.Module) -> dspy.Module:
        return self.optimizer.compile(module)
