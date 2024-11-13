
# Goals

This chatbot as it stands is pretty basic. For one, we want it to sound more like our client. We have already collected a few fake example conversations in training_data/conversations.json. We also want to improve it more generally.

1. **Improve Client Personality Emulation**  
   Use DSPy’s KNNFewShot optimizer (<https://dspy.ai/learn/optimization/optimizers/>) to make the chatbot’s responses reflect our client’s voice more authentically, based on examples in `conversations.json`.

    ***Thought Process:***

    > In the prompt, give the LLM a sample of a couple back and forth and try to copy that style. This is backed by some actual papers: Conversation Style Transfer using Few-Shot Learning: <https://arxiv.org/abs/2302.08362>

    > We can then use dspy metrics (say KNNs + some model) to see how far we possibly were from ideal (conversations.json). We will split the training data from the test samples. Question is, what embedding model should we use?!

    > Embedding Model: regular BERT might not be able to capture style transfer, albeit it might be close enough to start. We could also just start with regular llama, however [it might require a lil more work](https://hamel.dev/blog/posts/llm-judge/#step-5-build-your-llm-as-a-judge-iteratively) than this task requires for this time?  What embeddings should we use?! Is there a bert model that we can fetch so we can classify a little bit better on style?! Aparently StyleBERT is a thing but it doesn't seem to be offered by Together.

    > Confusignly, the [KNNFewShot cheat sheet](https://github.com/stanfordnlp/dspy/blob/6a3c3e7fb96b5a796af38ce2b4736c7b2741bccc/docs/docs/cheatsheet.md?plain=1#L466) doesn't show it taking a embedding model. What?! how is this supposed to work? I guess we will just use the included sentence_transformer stuff...

    > Ok, so here is the plan: We will load all of the examples but we just want to show the exchange and not the entire example. We want to show the LLM, for a given situation, a couple exhanges of a fan and creator. Ideally using the most relevant ones. That way the llm can stay within the same style.

    > I got this working. Vibe check is ok. Ideally we would want to verify by using a validation set. I am concerned that the KNN optimizer might be matching similarity based on the output response and not the input. Ideally I would like to debug this or as said above validate it but well see....

2. **Incorporate Context Awareness**  
   Introduce context awareness in a way that makes the chatbot more responsive to the timing and circumstances of each interaction. Examples might include awareness of the current time or the duration of a conversation.

3. **Topic Filtering**  
   Ensure the chatbot avoids discussing specific topics that may not be suitable. For this exercise, keep responses free of mentions of social media platforms (except OnlyFans) and interactions suggesting in-person meetings with fans.

4. **Further Product Enhancements**  
   Identify and implement an additional enhancement that you believe would improve the product experience.
