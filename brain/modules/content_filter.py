from dspy import Module, Predict
from brain.signatures.content_filter_responder import ContentFilterSignature


class ContentFilterModule(Module):

    def __init__(self):
        super().__init__()
        self.prog = Predict(ContentFilterSignature)

    def forward(self, message: str):
        """
        Filter message content for inappropriate topics according to the rules in
        ContentFilterSignature.
        """
        return self.prog(message=message)
