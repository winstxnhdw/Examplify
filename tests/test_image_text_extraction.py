# pylint: disable=missing-function-docstring,redefined-outer-name

from typing import Literal

from pytest import fixture
from PIL import Image
from fastapi import UploadFile

from server.features.extraction.extract_text import extract_texts_from_image

Text = Literal['Hello world!']


@fixture()
def image():
    with Image.open("./tests/assets/image.png") as img:
        image_bytes = img.tobytes()
    yield image_bytes

def test_image_text_extraction(image):
    print(extract_texts_from_image("image", image))
