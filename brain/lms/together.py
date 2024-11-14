import logging
import os

import backoff
import requests
from dsp.modules.hf import HFModel
from dsp.utils.settings import settings
from together import Together as TogetherClient

ERRORS = (Exception, requests.RequestException)

logger = logging.getLogger(__name__)


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    logger.warning(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details), )


class Together(HFModel):

    def __init__(
        self,
        model,
        api_base="",
        api_key=os.environ["TOGETHER_API_KEY"],
        **kwargs,
    ):
        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.model = model
        assert api_key, "Together API key is required"
        self.client = TogetherClient(api_key=api_key)
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
        logger.debug(f"Calling Together with prompt: {prompt}")
        try:
            response = self.client.completions.create(
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
            logger.debug(f"Response: {response}")
            completions = [response.choices[0].text]
            response = {
                "prompt": prompt,
                "choices": [{
                    "text": c
                } for c in completions],
            }
            return response
        except Exception as e:
            logging.error(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")
