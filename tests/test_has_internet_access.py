# pylint: disable=missing-function-docstring

from server.helpers.network.has_internet_access import has_internet_access


def test_has_internet_access():

    assert all([
        has_internet_access('winstxnhdw/openchat-3.5-ct2-int8'),
        has_internet_access('winstxnhdw/bge-base-en-v1.5-ct2')
    ]) is True
