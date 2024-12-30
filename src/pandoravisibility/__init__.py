import os


PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

from .visibility import Visibility  # noqa: E402, F401

__all__ = ["Visibility"]
