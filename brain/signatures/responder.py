from dspy import Signature, InputField, OutputField

from models import ChatHistory


class Responder(Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You are deciding on what your response message should be.
    """

    chat_history: ChatHistory = InputField(desc="the chat history")
    response: str = OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )
