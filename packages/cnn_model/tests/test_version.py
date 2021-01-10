import pytest


def test_version_returned():
    from cnn_model import __version__ as _version

    assert _version != None

