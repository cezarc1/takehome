from models import ChatMessage, ChatHistory, LabeledChatHistory
import dspy
from lms.together import Together

from modules.chatter import ChatterModule

lm = Together(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
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

dspy.settings.configure(lm=lm)

training_examples = LabeledChatHistory.load_labeled_histories()
user_chat_history = ChatHistory()
chatter = ChatterModule(examples=training_examples)
while True:
    # Get user input
    user_input = input("You: ")

    # Append user input to chat history
    user_chat_history.messages.append(
        ChatMessage(
            from_creator=False,
            content=user_input,
        ), )

    # Send request to endpoint
    response = chatter(chat_history=user_chat_history).output

    # Append response to chat history
    user_chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
        ), )
    # Print response
    print()
    print("<Debug>")
    print("Prompt:", lm.inspect_history(n=1))
    print("</Debug>")
    print("Response:", response)

    print()
