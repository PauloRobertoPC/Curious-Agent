import cv2
import numpy as np
import copy
from manim import *

class VideoMobject(ImageMobject):
    """
    Classe customizada para reproduzir vídeos .mp4 dentro do Manim CE.
    Otimizada para performance redimensionando os frames e usando conversão nativa RGBA.
    """
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)
        ret, frame = self.cap.read()
        
        if ret:
            # Redimensionar o frame para melhorar drasticamente a performance de renderização
            frame = cv2.resize(frame, (320, 240))
            # Conversão nativa e rápida do OpenCV direto para RGBA
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            print(f"Erro: Não foi possível carregar o vídeo {filename}.")
            rgba_frame = np.zeros((240, 320, 4), dtype=np.uint8)
            
        super().__init__(rgba_frame, **kwargs)
        self.add_updater(self.update_frame)

    def update_frame(self, mobj, dt):
        if hasattr(self, 'cap') and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Mesmo redimensionamento e conversão rápida no loop
                frame = cv2.resize(frame, (320, 240))
                rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                mobj.pixel_array = rgba_frame
            else:
                # Reinicia o vídeo ao final para criar o Loop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k == 'cap':
                setattr(result, k, cv2.VideoCapture(self.filename))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        
        return result


def glaucoma(self):
    # 1. Título
    title = Text("Precariedade Sensorial: O Glaucoma", font_size=36, weight=BOLD).to_edge(UP, buff=0.3)
    
    explicacao = Paragraph(
        "A perda progressiva da visão periférica.",
        "Testando a adaptabilidade do agente à degradação.",
        alignment="center",
        font_size=24
    ).next_to(title, DOWN, buff=0.2)

    self.play(FadeIn(title), FadeIn(explicacao))
    self.next_slide()

    # 2. Carregando os Vídeos (tamanho reduzido para caberem os 4)
    vid_width = 3.8
    vid_0 = VideoMobject("assets/glaucoma-0.mp4").scale_to_fit_width(vid_width)
    vid_50 = VideoMobject("assets/glaucoma-50.mp4").scale_to_fit_width(vid_width)
    vid_100 = VideoMobject("assets/glaucoma-100.mp4").scale_to_fit_width(vid_width)
    vid_200 = VideoMobject("assets/glaucoma-200.mp4").scale_to_fit_width(vid_width)

    # 3. Criando as Legendas ajustadas
    lbl_0 = Text("Força 0 (Visão Normal)", font_size=20).next_to(vid_0, DOWN, buff=0.1)
    lbl_50 = Text("Força 50", font_size=20).next_to(vid_50, DOWN, buff=0.1)
    lbl_100 = Text("Força 100", font_size=20).next_to(vid_100, DOWN, buff=0.1)
    lbl_200 = Text("Força 200 (Severo)", font_size=20).next_to(vid_200, DOWN, buff=0.1)

    # UTILIZANDO Group EM VEZ DE VGroup PARA IMAGENS/VÍDEOS
    g_0 = Group(vid_0, lbl_0)
    g_50 = Group(vid_50, lbl_50)
    g_100 = Group(vid_100, lbl_100)
    g_200 = Group(vid_200, lbl_200)

    # 4. Organizando em uma Grade Compacta (Também utilizando Group)
    grid = Group(g_0, g_50, g_100, g_200).arrange_in_grid(rows=2, cols=2, buff=0.4)
    
    # Posiciona a grade inteira logo abaixo do título
    grid.next_to(title, DOWN, buff=0.3)

    self.play(
        FadeOut(explicacao),
        FadeIn(grid)
    )
    
    # 5. Tempo de Reprodução
    self.wait(6) 
    
    self.next_slide()

    # Limpa a tela para a próxima cena
    self.play(*[FadeOut(mob) for mob in self.mobjects])
