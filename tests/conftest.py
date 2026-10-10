import pytest
import requests
from PIL import Image


def get_image(url: str) -> Image.Image:
    with requests.get(
        url, stream=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 60)
    ) as resp:
        resp.raise_for_status()
        with Image.open(resp.raw) as image:
            return image.copy()


@pytest.fixture(scope="session")
def sample_image():
    return get_image("https://i.imgur.com/E5OVXvn.png")


@pytest.fixture(scope="session")
def sample_image2():
    return get_image("https://i.imgur.com/px5UT7T.png")
