import os


PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

from .visibility import Visibility

__all__ = ["Visibility"]
