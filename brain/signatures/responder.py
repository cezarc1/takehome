from typing import Optional

from dspy import InputField, OutputField, Signature
from models import ChatHistory


class Responder(Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a FAN.
    You are deciding on what your response message should be.
    Your response should be in the same voice, tone, language, length
    and especially style as the previous messages, if any. Only use the 1st
    person and never include any other details, including your thoughts,
    reasoning, etc when writing your message back to the fan.
    """

    chat_history: ChatHistory = InputField(desc="the chat history.")
    img_base64: Optional[str] = InputField(
        desc="the base64 encoded image sent to you by the fan, if any. This "
        "is only sent if the fan sent an image to you and may not always be "
        "present.",
        is_image=True)
    response: str = OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )
