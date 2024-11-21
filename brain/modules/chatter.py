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
            model_name_or_path="StyleDistance/styledistance",
            normalize_embeddings=True)
        self.responder = ResponderModule()
        if use_filter:
            self.content_filter = ContentFilterModule()
        else:
            self.content_filter = None

    def compile(self):
        logging.info(
            f"Compiling {self.responder} with KNNFewShot optimizer using "
            f"{len(self.dspy_examples)} training examples.")
        optimizer = KNNFewShot(
            k=3,
            max_rounds=3,
            trainset=self.
            dspy_examples,  # type: ignore KNNFewShot uses dsp.Example... instead of dspy.Example. Why?!
            vectorizer=self.vectorizer,
            metric=self.similarity_match_metric,
        )
        # Note that we did not compile the content filter, ideally we might.
        self.responder = optimizer.compile(self.responder)

    def forward(
        self,
        chat_history: ChatHistory,
        img_base64: Optional[str] = None,
    ):
        initial_response = self.responder(
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
            f"Evaluating KNN module with {len(self.dspy_examples)} training "
            f"examples on {str(self.responder)}...")
        eval = Evaluate(devset=self.dspy_examples,
                        num_threads=5,
                        metric=self.similarity_match_metric)
        eval_score: float = eval(self.responder)  # type: ignore
        return {"avg_similarity_score": eval_score / 100.0}

    def similarity_match_metric(self,
                                example: Example,
                                pred: Prediction,
                                trace: object = None):
        """Hybrid metric combining F1 score and vector similarity."""
        f1_score = self.f1_metric(example, pred, trace)
        vec_sim_score = self.vector_similarity(example, pred, trace)

        if trace is not None:
            return vec_sim_score and f1_score

        # These weights were chosen arbitrarily but they seem to work well.
        f1_weight = 0.3
        vec_sim_weight = 0.7
        return (f1_score * f1_weight) + (vec_sim_score * vec_sim_weight)

    def vector_similarity(self,
                          example: Example,
                          pred: Prediction,
                          trace: object = None):
        """
        This metric is used to determine if the predicted response is a good 
        match for the example response. This calculates the cosine similarity
        between the predicted response and the example response.
        """
        assert (type(example.response) is str)
        predicted_vector = self.vectorizer([normalize_text(pred.response)
                                            ]).astype(np.float32)
        example_vector = self.vectorizer([normalize_text(example.response)
                                          ]).astype(np.float32)
        score = np.dot(example_vector, predicted_vector.T).squeeze()

        if trace is None:
            return score.item()
        return score.item(
        ) >= 0.4  # TODO: determine if this is a good threshold

    @staticmethod
    def f1_metric(example: Example, pred: Prediction, trace: object = None):
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
            return f1 >= 0.03  # TODO: determine if this is a good threshold
