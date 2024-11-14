from dsp.modules.hf import HFModel
import os
import requests
import backoff
from dsp.utils.settings import settings

from together import Together

client = Together()

ERRORS = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class Together(HFModel):
    def __init__(
        self,
        model,
        api_base="",
        api_key="",
        **kwargs,
    ):
        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.model = model

        self.kwargs = {**kwargs}

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
    )
    def _generate(self,
                  prompt,
                  use_chat_api=False,
                  image_base64=None,
                  **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        # stop = kwargs.get("stop")

        try:
            response = client.completions.create(
                prompt=prompt,
                model=self.model,
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
                repetition_penalty=kwargs.get("repetition_penalty"),
                stop=kwargs.get("stop"),
                image_base64=kwargs.get("image_base64"),
                stream=False,
            )
            completions = [response.choices[0].text]
            response = {
                "prompt": prompt,
                "choices": [{
                    "text": c
                } for c in completions],
            }
            return response

        except Exception as e:
            if response:
                print(f"resp_json:{response.json}")
            print(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")
