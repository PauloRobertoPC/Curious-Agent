from manim import *
from manim_slides import Slide

from code.title import title
from code.reinforcement_learning import reinforcement_learning
from code.deep_reinforcement_learning import deep_reinforcement_learning
from code.ppo import ppo
from code.ambient import ambient
from code.observacao import observacao
from code.acao import acao
from code.recompensa_extrinsica import recompensa_extrinsica
from code.problema_recompensa import problema_recompensa
from code.turing import turing
from code.chines import chines
from code.ponte import ponte

class Presentation(Slide):
    skip_reversing = False
    def construct(self):
        # title(self)
        # reinforcement_learning(self)
        # deep_reinforcement_learning(self)
        # ppo(self)
        # ambient(self)
        # observacao(self)
        # acao(self)
        # recompensa_extrinsica(self)
        # problema_recompensa(self)
        # turing(self)
        chines(self)
        ponte(self)

