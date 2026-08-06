from manim import *
from manim_slides import Slide

from code.title import title

class Presentation(Slide):
    skip_reversing = False
    def construct(self):
        title(self)

