import logging

from dspy import Module, Example
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import KNNFewShot
from models import LabeledChatHistory
import dsp


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
            f"Evaluating KNN module with {len(self.dspy_examples)} training examples on {str(module)}..."
        )

        def answer_exact_match(example: Example, pred, frac=0.75):
            assert (type(example.response) is str
                    or type(example.response) is list)

            if type(example.response) is str:
                return dsp.answer_match(pred.response, [example.response],
                                        frac=frac)
            else:  # type(example.response) is list
                return dsp.answer_match(pred.response,
                                        example.response,
                                        frac=frac)

        evaluate_on_examples = Evaluate(devset=self.dspy_examples,
                                        num_threads=1,
                                        display_progress=True,
                                        display_table=True,
                                        provide_traceback=False,
                                        return_outputs=False,
                                        metric=answer_exact_match)
        return evaluate_on_examples(module)
