import base64
import re
from io import BytesIO
from typing import Optional

import requests
from PIL import Image


def extract_image_from_text(text: str) -> Optional[str]:
    """Extract base64 encoded image from text containing a URL and return just
    the base64 encoded image if a url is found.

    Args:
        text: Input text that may contain a URL
        
    Returns:
        Base64 encoded image if a url is found, otherwise None
    """
    if "http" not in text:
        return None

    url_match = re.search(r"http[^\s]+", text)
    if not url_match:
        return None

    image_url = url_match.group(0)
    image = fetch_image_from_url(image_url)
    image_format = image_url.split(".")[-1]
    image_b64 = image_to_base64(image, image_format)
    return image_b64


def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
