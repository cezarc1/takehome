import logging
from datetime import datetime

from dspy import settings
from image_utils import extract_image_from_text
from lms.together import Together
from models import ChatHistory, ChatMessage, LabeledChatHistory
from modules.chatter import ChatterModule

logging.basicConfig(level=logging.INFO)

lm = Together(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.2,
    stop=[
        "<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"
    ],
    # stop=["\n", "\n\n"],
)

settings.configure(lm=lm)

training_examples = LabeledChatHistory.load_labeled_histories()
chatter = ChatterModule(examples=training_examples)
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
            image_base64=image_base64,
        ), )

    # Send request to endpoint
    response = chatter(chat_history=user_chat_history).response

    # Append response to chat history
    user_chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
            timestamp=datetime.now(),
        ), )
    # Print response
    print()
    # print("<Debug>")
    # print(
    #     "Prompt:", lm.inspect_history(n=2)
    # )  # we send two messages at a time (1 for the response and another for the content filter)
    # print("</Debug>")
    print("Response:", response)

    print()
