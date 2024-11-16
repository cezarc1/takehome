import logging
import os
from datetime import datetime

from dspy import settings
from image_utils import extract_image_from_text
from lms.together import Together
from models import ChatHistory, ChatMessage, LabeledChatHistory
from modules.chatter import ChatterModule

logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'ERROR')))
logger = logging.getLogger(__name__)

together_api_key = os.environ.get("TOGETHER_API_KEY")
if not together_api_key:
    raise ValueError(
        "TOGETHER_API_KEY environment variable is not set. Create a .env file in the root project with the key"
    )
EVAL_MODE = os.environ.get("EVAL_MODE") == "true"

print("Loading model...")

lm = Together(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    api_key=together_api_key,
    temperature=0.5,
    max_tokens=1000,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.2,
    stop=[
        "<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"
    ],
)

settings.configure(lm=lm)

training_examples = LabeledChatHistory.load_labeled_histories()
logger.info(f"Loaded {len(training_examples)} training examples")
logger.info("Loading ChatterModule...")
chatter = ChatterModule(examples=training_examples)
logger.info("ChatterModule loaded")
if EVAL_MODE:
    logger.info("Evaluating ChatterModule...")
    logger.info(f"ChatterModule evaluation results: {chatter.evaluate()}")
user_chat_history = ChatHistory()
while True:
    # Get user input
    user_input = input("You: ")

    # if the user input contains a url, assume its an image and try to fetch it
    image_base64 = extract_image_from_text(user_input)
    if image_base64:
        logging.info("Image found in user input")

    # Append user input to chat history
    user_chat_history.messages.append(
        ChatMessage(
            from_creator=False,
            content=user_input,
            timestamp=datetime.now(),
            image_base64=
            image_base64,  # we only add the image here for bookkeeping
        ), )

    logger.debug(f"Last 2 prompts: {lm.inspect_history(n=2)}")

    # Send request to endpoint
    print("Generating response...")
    response = chatter(
        chat_history=user_chat_history,
        img_base64=image_base64,
    ).response

    # Append response to chat history
    user_chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
            timestamp=datetime.now(),
        ), )
    # Print response
    print()
    print("Creator: ", response)
    print()
