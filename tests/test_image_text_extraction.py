# pylint: disable=missing-function-docstring,redefined-outer-name

from pytest import fixture
from PIL import Image
from fastapi import UploadFile

from server.features.extraction.extract_from_image import (
    extract_text_from_image, 
    extract_document_from_image
)

@fixture()
def image():
    with Image.open("./tests/assets/image.png") as img:
        yield img

def test_image_document_extraction(image):
    assert len(extract_document_from_image("image", image).sections) > 0

def test_image_text_extraction(image):
    assert len(extract_text_from_image(image)) > 0