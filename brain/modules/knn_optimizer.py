import logging

from dspy import Module, Example
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory
import dsp


class KNNOptimizer():

    def __init__(self, examples: list[LabeledChatHistory], k: int = 5):
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
            f"Evaluating KNN module with {len(self.dspy_examples)} training examples on {str(module)}..."
        )

        evaluate_on_examples = Evaluate(
            devset=self.dspy_examples,
            num_threads=1,
            display_progress=True,
            display_table=True,
            provide_traceback=False,
            return_outputs=False,
            # TODO: measure using vector similarity also
            metric=self.answer_similarity_match_f1)
        return evaluate_on_examples(module)

    @staticmethod
    def answer_similarity_match_f1(example: Example,
                                   pred,
                                   trace: object = None):
        assert (type(example.response) is str)
        f1 = dsp.F1(pred.response, [example.response])
        return f1 >= 0.05
