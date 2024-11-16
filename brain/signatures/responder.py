from dspy import InputField, OutputField, Signature
from models import ChatHistory


class Responder(Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a FAN.
    You are deciding on what your response message should be.
    Your response should be in the same voice and tone as the previous
    messages, if any.
    """

    chat_history: ChatHistory = InputField(desc="the chat history")
    response: str = OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )
