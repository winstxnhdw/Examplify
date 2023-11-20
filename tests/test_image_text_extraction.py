# pylint: disable=missing-function-docstring,redefined-outer-name

from pytest import fixture
from PIL import Image
from fastapi import UploadFile

from server.features.extraction.extract_text import extract_texts_from_image


@fixture()
def image():
    with Image.open("./tests/assets/image.png") as img:
        yield img.tobytes()


def test_image_text_extraction(image):
    print(extract_texts_from_image("image", image))
