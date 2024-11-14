from dspy import InputField, OutputField, Signature


class ContentFilterSignature(Signature):
    """Check if a message contains inappropriate content.

    Outputs:
    - is_safe: Whether the message is safe to send
    - reasoning: Reasoning behind the decision
    - filtered_message: Modified message if needed, or original if safe

    Rules:
    - No social media mentions (except OnlyFans)
    - No suggestions of in-person meetings
    """

    message: str = InputField(
        desc="Message to check for inappropriate content")
    is_safe: bool = OutputField(
        desc="Whether the message is safe to send (True/False). Only "
        "respond with False if the message is not safe or True if it is. "
        "Do not respond with any")
    reasoning: str = OutputField(
        desc="Explanation of why the message was flagged or approved")
    filtered_message: str = OutputField(
        desc="Modified message if needed, or original if safe")
