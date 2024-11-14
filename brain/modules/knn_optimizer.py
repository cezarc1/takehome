import logging

from dspy import Module
from dspy.evaluate import answer_passage_match
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory


class KNNOptimizer():

    def __init__(self, examples: list[LabeledChatHistory], k: int = 3):
        super().__init__()
        self.labeled_examples = examples
        self.dspy_examples = [
            example.to_dspy_example() for example in self.labeled_examples
        ]
        self.optimizer = KNNFewShot(
            k,
            trainset=self.
            dspy_examples,  # type: ignore KNNFewShot uses dsp.Example... instead of dspy.Example
        )

    def compile_module(self, module: Module) -> Module:
        logging.info(
            f"Compiling KNN module with {len(self.labeled_examples )} training examples."
        )
        return self.optimizer.compile(module)

    def evaluate(self, module: Module):
        logging.info(
            f"Evaluating KNN module with {len(self.dspy_examples)} training examples..."
        )
        evaluate_on_valset = Evaluate(devset=self.dspy_examples,
                                      num_threads=1,
                                      display_progress=True,
                                      display_table=True,
                                      provide_traceback=True,
                                      return_outputs=True)
        return evaluate_on_valset(module, answer_passage_match)
