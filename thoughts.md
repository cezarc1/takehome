# Goals
>
> This chatbot as it stands is pretty basic. For one, we want it to sound more like our client. We have already collected a few fake example conversations in training_data/conversations.json. We also want to improve it more generally.
>
> 1. **Improve Client Personality Emulation**  
>    Use DSPy's KNNFewShot optimizer (<https://dspy.ai/learn/optimization/optimizers/>) to make the chatbot's responses reflect our client's voice more authentically, based on examples in `conversations.json`.
>
> ***Thought Process:***

In the prompt, give the LLM a sample of a couple back and forth and try to copy that style. This is backed by some actual papers: Conversation Style Transfer using Few-Shot Learning: <https://arxiv.org/abs/2302.08362>

We can then use DSPy metrics (say KNNs + some model) to see how far we possibly were from ideal (conversations.json). We will split the training data from the test samples. Question is, what embedding model should we use?!

Embedding Model: regular BERT might not be able to capture style transfer, albeit it might be close enough to start. We could also just start with regular llama, however [it might require a lil more work](https://hamel.dev/blog/posts/llm-judge/#step-5-build-your-llm-as-a-judge-iteratively) than this task requires for this time?  What embeddings should we use?! Is there a BERT model that we can fetch so we can classify a little bit better on style?! Apparently StyleBERT is a thing but it doesn't seem to be offered by Together.

Confusingly, the [KNNFewShot cheat sheet](https://github.com/stanfordnlp/dspy/blob/6a3c3e7fb96b5a796af38ce2b4736c7b2741bccc/docs/docs/cheatsheet.md?plain=1#L466) doesn't show it taking a embedding model. What?! how is this supposed to work? I guess we will just use the included sentence_transformer stuff... Ahh It just uses all-MiniLM-L6-v2 as the default embedding model.

Ok, so here is the plan: We will load all of the examples but we just want to show the exchange and not the entire example. We want to show the LLM, for a given situation, a couple exchanges of a fan and creator. Ideally using the most relevant ones. That way the llm can stay within the same style.

I got this working. Vibe check is ok. Ideally we would want to verify by using a validation set. I am concerned that the KNN optimizer might be matching similarity based on the output response and not the input. Ideally I would like to debug this or as said above validate it but well see.... I am NOT sure that this is doing it correctly.

Future improvements: We could sample sub-sequences of the chat history to see if that helps.
Eval: we should evaluate the responses based on a couple more metrics (style, topic, etc.)

Eval using the same 10 samples that we used for training. Using F1 (bag of words) as bootstrap metric on `Llama-3.2-90B-Vision-Instruct-Turbo`:

| k | f1 threshold | Average Metric |
|---|-------------|----------------|
| 1 | >= 0.10 | 3/10 (30.0%) |
| 1 | >= 0.05 | 6/10 (60.0%) |
| 3 | >= 0.05 | 8/10 (80.0%) |
| 3 | >= 0.10 | 5/10 (50.0%) |
| 5 | >= 0.10 | 6/10 (60.0%) |
| 5 | >= 0.05 | 10/10 (100.0%) |

Using Vector similarity as bootstrap metric, using  from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):

| k | vector similarity threshold | Average Metric |
|---|---------------------------|----------------|
| 5 | >= 0.4 | 4.435821399092674 / 10 (44.4%) |
| 1 | >= 0.4 | 4.760176405310631 / 10 (47.6%) |
| 1 | >= 0.3 | 4.4036207646131516 / 10 (44.0%) |

In the end, I went with the F1 metric as the bootstrap metric, with the following parameters: k = 1, f1 >= 0.05, vec_sim >= 0.4.

Best eval run on `Llama-3.2-90B-Vision-Instruct-Turbo` using these parameters was: {'avg_f1_score': 0.35259999999999997, 'avg_vector_similarity_score': 0.5638000000000001}

Also, worth noting that the best eval run on `Meta-Llama-3.1-405B-Instruct-Turbo` using these parameters was: {'avg_f1_score': 0.10980000000000001, 'avg_vector_similarity_score': 0.4172}

Switched to using [StyleDistance/styledistance](https://huggingface.co/StyleDistance/styledistance) as the embedding model for calculating vector similarity, using `Llama-3.2-90B-Vision-Instruct-Turbo`:

Using StyleDistance/styledistance as embedding model with `Llama-3.2-90B-Vision-Instruct-Turbo` as the llm model:

| k | bootstrap_max_rounds | thresholds | F1 Score | Vector Similarity |
|---|---------------------|------------|----------|------------------|
| 1 | 1 | vec_sim >= 0.4, f1 > 0.05 | 0.0640 | 0.8644 |
| 2 | 1 | vec_sim >= 0.4, f1 > 0.05 | 0.0973 | 0.8883 |
| 3 | 1 | vec_sim >= 0.4, f1 > 0.05 | 0.0561 | 0.8738 |
| 6 | 1 | vec_sim >= 0.4, f1 > 0.05 | 0.0862 | 0.8652 |
| 3 | 3 | vec_sim >= 0.4, f1 > 0.05 | 0.0593 | 0.8984 |

****🔧 Final Configuration:****

I ended up using the `vec_sim` metric for bootstrap metric, with the following parameters: `k = 3`, `vec_sim >= 0.4`, `f1 > 0.05`.
>
> 2. **Incorporate Context Awareness**  
>    Introduce context awareness in a way that makes the chatbot more responsive to the timing and circumstances of each interaction. Examples might include awareness of the current time or the duration of a conversation.

I added the time gap between live conversations but not between messages for the llm to respond to. There could be other things, that would be relevant, like holidays or things outside of the memory. Or perhaps something specific about the fan (their birthday, etc.)

> 3. **Topic Filtering**  
>    Ensure the chatbot avoids discussing specific topics that may not be suitable. For this exercise, keep responses free of mentions of social media platforms (except OnlyFans) and interactions suggesting in-person meetings with fans.

Added bare bones filtering. Logic is that we detect if we need to filter the response and if we do we return the filtered response, otherwise we return the response as usual. We also get a reasoning back, which ideally we should log. Obviously this is inefficient, especially if we send the filter response to the same model. Ideally we would have a separate, smaller model for detecting if we need to filter.

> 4. **Further Product Enhancements**  
>    Identify and implement an additional enhancement that you believe would improve the product experience.

***Initial thoughts:***

1. Personalization: Would be nice to have the creator be able to remember things about the fan. Preferences, birthdays, things like that. This could also blow up in our face if we are too personal. Initial thoughts are to use a KG to store and retrieve this information. We could use a simple Signature for extracting out useful information to remember. The signature is simple but integrating a KG is somehow bound to be a headache to integrate with dspy? Took a look at the retrieval modules and there is a [neo4j](https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/neo4j_rm.py) one....

2. Async messaging: Would be nice to have the ability to send messages asynchronously. For example, if the fan has not messaged for a while, the creator could send a message reminding them of something? promo?! something personal?

3. Per, creator personalization. Unsure if this is the whole point of the chatbot. i.e. for each creator we have our own training data.
4. Use a vector store for the KNN stuff. Wasted work otherwise.
5. More training data. Maybe we can synthetically generate some?
6. understand images and respond to them.

In the end I chose to add photo support. I think it is a nice feature. And it was relatively easy to add. If we see a http in the text, we assume that its an image, and we try to fetch the image and return it.

## How to run this locally

1. Create a `.env` file in the root of the project and set the `TOGETHER_API_KEY` environment variable. Sign up for an API key [here](https://api.together.xyz/), and then grab your api key from [here](https://api.together.xyz/settings/api-keys).
2. Go into VSCode or Cursor Debug panel and select either `Python: local_chat`or `Python: local_chat DEBUG` for chatting or debugging the model, `Python: local_chat EVAL` for *evaluating* the model, lastly `Python: Attach To Jupyter` for attaching a debugger to a running jupyter notebook. *NOTE: The launch config should automatically handle installing the poetry deps.*
