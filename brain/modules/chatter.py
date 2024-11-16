import logging
from typing import Optional

import dsp
import numpy as np
from dsp.utils.metrics import normalize_text
from dspy import Example, Module, Prediction
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import KNNFewShot
from models import ChatHistory, LabeledChatHistory

from .content_filter import ContentFilterModule
from .responder import ResponderModule

logger = logging.getLogger(__name__)


class ChatterModule(Module):

    def __init__(self,
                 examples: list[LabeledChatHistory],
                 use_filter: bool = True):
        super().__init__()
        self.examples = examples
        self.dspy_examples = [
            example.to_dspy_example() for example in self.examples
        ]
        self.vectorizer = dsp.SentenceTransformersVectorizer(
            normalize_embeddings=True)
        self.optimizer = KNNFewShot(
            k=1,
            max_rounds=1,
            trainset=self.
            dspy_examples,  # type: ignore KNNFewShot uses dsp.Example... instead of dspy.Example. Why?!
            vectorizer=self.vectorizer,
            metric=self.get_similarity_match_vector_metric(self.vectorizer),
        )
        logging.info(
            f"Compiling ResponderModule with KNNFewShot optimizer using {len(self.dspy_examples)} training examples."
        )
        self.compiled_responder = self.optimizer.compile(ResponderModule())
        if use_filter:
            self.content_filter = ContentFilterModule()

    def forward(
        self,
        chat_history: ChatHistory,
        img_base64: Optional[str] = None,
    ):
        initial_response = self.compiled_responder(
            chat_history=chat_history,
            img_base64=img_base64,
        )
        if self.content_filter:
            filtered = self.content_filter(message=initial_response.response)
            if not filtered.is_safe:
                logger.warning(
                    f"Message detected as NOT SAFE: {initial_response.response}.\n"
                    f"REASON: {filtered.reasoning}\n"
                    f"new filtered message: {filtered.filtered_message}.")
                return Prediction(response=filtered.filtered_message)
        return initial_response

    def evaluate(self) -> dict[str, float]:
        # Note that we are evaluating this module with the same examples that
        # we trained on. This is less than ideal, but we only have 10 samples
        # so... ¯\_(ツ)_/¯.
        logging.info(
            f"Evaluating KNN module using F1 score with {len(self.dspy_examples)} training examples on {str(self.compiled_responder)}..."
        )
        f1_eval = Evaluate(devset=self.dspy_examples,
                           num_threads=5,
                           metric=self.similarity_match_f1_metric)
        f1_eval_score: float = f1_eval(self.compiled_responder)  # type: ignore

        logging.info(
            f"Evaluating KNN module using vector similarity with {len(self.dspy_examples)} training examples on {str(self.compiled_responder)}..."
        )
        vector_sim_eval = Evaluate(
            devset=self.dspy_examples,
            num_threads=5,
            metric=self.get_similarity_match_vector_metric(self.vectorizer))
        vector_eval_score: float = vector_sim_eval(
            self.compiled_responder)  # type: ignore
        return {
            "avg_f1_score": f1_eval_score / 100.0,
            "avg_vector_similarity_score": vector_eval_score / 100.0
        }

    @staticmethod
    def get_similarity_match_vector_metric(
            vectorizer: dsp.BaseSentenceVectorizer):

        def answer_similarity_match_vector(example: Example,
                                           pred: Prediction,
                                           trace: object = None):
            """
            This metric is used to determine if the predicted response is a good 
            match for the example response. This calculates the cosine similarity
            between the predicted response and the example response.
            """
            assert (type(example.response) is str)
            predicted_vector = vectorizer([normalize_text(pred.response)
                                           ]).astype(np.float32)
            example_vector = vectorizer([normalize_text(example.response)
                                         ]).astype(np.float32)
            score = np.dot(example_vector, predicted_vector.T).squeeze()
            if trace is None:
                # we are in eval or optimization mode return the score (float)
                return score.item()
            else:
                # we are in bootstrapping mode; return a boolean to
                # determine if the pred is a good match
                return score.item(
                ) >= 0.4  # TODO: determine if this is a good threshold

        return answer_similarity_match_vector

    @staticmethod
    def similarity_match_f1_metric(example: Example,
                                   pred: Prediction,
                                   trace: object = None):
        """
        This metric is used to determine if the predicted response is a good 
        match for the example response. This calculates the F1 score between
        the predicted response and the example response using a bag of words
        approach.
        """
        assert (type(example.response) is str)
        f1 = dsp.F1(pred.response, [example.response])
        if trace is None:
            # we are in eval or optimization mode return the score (float)
            return f1
        else:
            # we are in bootstrapping mode; return a boolean to
            # determine if the pred is a good match
            return f1 >= 0.05
