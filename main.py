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
from code.autopoiese import autopoiese
from code.acoplamento_estrutural import acoplamento_estrutural
from code.producao_sentido import producao_sentido
from code.mundo_ambiente import mundo_ambiente
from code.adaptatividade import adaptatividade
from code.ia_enativa import ia_enativa
from code.ponte_pratica import ponte_pratica
from code.glaucoma import glaucoma
from code.motivacao_recompensa import motivacao_recompensa
from code.rnd import rnd
from code.grafico_rnd import grafico_rnd
from code.intrinsic_reward import intrinsic_reward
from code.episode_length import episode_length
from code.trajetorias import trajetorias
from code.gradcam import gradcam
from code.fim import fim

class Presentation(Slide):
    skip_reversing = False
    def construct(self):
        title(self)
        reinforcement_learning(self)
        ambient(self)
        observacao(self)
        acao(self)
        deep_reinforcement_learning(self)
        ppo(self)
        recompensa_extrinsica(self)
        problema_recompensa(self)
        turing(self)
        chines(self)
        ponte(self)
        autopoiese(self)
        acoplamento_estrutural(self)
        producao_sentido(self)
        mundo_ambiente(self)
        adaptatividade(self)
        ia_enativa(self)
        ponte_pratica(self)
        glaucoma(self)
        motivacao_recompensa(self)
        rnd(self)
        grafico_rnd(self, "assets/50_CIRCLE_0000200704.pt", [172, 173], 475)
        intrinsic_reward(self)
        episode_length(self)
        trajetorias(self)
        gradcam(self)
        fim(self)
