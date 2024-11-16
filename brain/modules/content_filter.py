from dspy import Module, OutputField, TypedChainOfThought

from brain.signatures.content_filter_responder import ContentFilterSignature


class ContentFilterModule(Module):

    def __init__(self):
        super().__init__()
        reasoning = OutputField(
            prefix="Reasoning: Let's think step by step to decide on the "
            "filtering decision.",
            desc="Reasoning for the filtering decision.",
        )
        # Use a TypedPredictor because the regular predictor doesn't actually work
        # regardless of the warnings.
        self.prog = TypedChainOfThought(ContentFilterSignature,
                                        reasoning=reasoning)

    def forward(self, message: str):
        """
        Filter message content for inappropriate topics according to the rules
        in ContentFilterSignature.
        """
        return self.prog(message=message)
