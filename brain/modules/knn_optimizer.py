import logging

from dspy import Module
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory


class KNNOptimizerModule():

    def __init__(self, examples: list[LabeledChatHistory], k: int = 3):
        super().__init__()
        self.training_examples = [
            example.to_dspy_example() for example in examples
        ]
        # split into train and validation sets. just pick first 80% as train set
        self.train_examples = self.training_examples[:int(
            0.8 * len(self.training_examples))]
        self.val_examples = self.training_examples[
            int(0.8 * len(self.training_examples)):]
        self.optimizer = KNNFewShot(
            k,
            # for some reason KNNFewShot uses dsp.Example vs dspy.Example :shrug:
            trainset=self.train_examples,  # type: ignore
        )

    def compile_module(self, module: Module) -> Module:
        logging.info(
            f"Compiling KNN module with {len(self.train_examples)} training examples and {len(self.val_examples)} validation examples"
        )
        return self.optimizer.compile(module,
                                      trainset=self.train_examples,
                                      valset=self.val_examples)
