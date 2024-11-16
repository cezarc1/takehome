from dspy import InputField, OutputField, Signature


class ContentFilterSignature(Signature):
    """Check if a message contains inappropriate content as defined by the
    rules below.

    Filtering Rules:
    - No social media mentions or references besides OnlyFans. i.e. no
      references to Instagram, Twitter or X, TikTok, etc. This is not allowed.
    - No suggestions of in-person meetings. This is not allowed.
    - Receiving or sending images or links is ok. This is allowed.
    - Adult content is ok, this is OnlyFans. This is allowed.
    - Threats, harassment, or hate speech is not allowed. This is not allowed.
    - Use common sense and good judgement.
    """

    message: str = InputField(
        desc="Message to check for inappropriate content as defined by the "
        "rules.")
    is_safe: bool = OutputField(
        desc="Whether the message is safe to send (True/False). Only "
        "respond with False if the message is not safe or flagged or True if "
        "it is safe or not flagged. This should always be filled in.")
    reasoning: str = OutputField(
        desc="Explanation of why the message was flagged or approved as "
        "defined by the rules. If the message was flagged then is_safe should "
        "be False. This should always be filled in.")
    filtered_message: str = OutputField(
        desc="Modified message if needed, or original if safe as defined by "
        "the rules. If the message was flagged, i.e. is_safe is False, then "
        "this should be the modified message. This should always be filled in."
    )
