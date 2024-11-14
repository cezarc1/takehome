from dspy import Module, TypedPredictor
from brain.signatures.content_filter_responder import ContentFilterSignature


class ContentFilterModule(Module):

    def __init__(self):
        super().__init__()
        # Use a TypedPredictor because the regular predictor doesn't actually work
        # regardless of the warnings.
        self.prog = TypedPredictor(ContentFilterSignature, explain_errors=True)

    def forward(self, message: str):
        """
        Filter message content for inappropriate topics according to the rules
        in ContentFilterSignature.
        """
        return self.prog(message=message)
