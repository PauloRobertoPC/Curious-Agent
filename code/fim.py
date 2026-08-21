from manim import *

def fim(self):
    # Limpa a tela de resíduos do slide anterior
    self.remove(*self.mobjects)

    # ==========================================
    # PARTE 1: Agradecimento e Título
    # ==========================================
    
    # Mensagem principal
    title = Text("Obrigado!", font_size=72, weight=BOLD, color=WHITE)
    title.shift(UP * 1.5)

    # O título da sua dissertação cimentando a apresentação
    subtitle = Text("Precariedade é Tudo Que Você Precisa", font_size=36, color=GREEN_C, weight=BOLD)
    subtitle.next_to(title, DOWN, buff=0.5)

    self.play(Write(title), run_time=1.5)
    self.play(FadeIn(subtitle, shift=UP*0.3))

    # ==========================================
    # PARTE 2: O Agente Enativo Final
    # ==========================================
    
    # Recriamos o agente num tamanho menor para o centro da tela
    body = RegularPolygon(n=6, color=BLUE, stroke_width=4).scale(0.8)
    core = Circle(radius=0.25, color=RED, fill_color=RED, fill_opacity=0.6)
    
    agent = VGroup(body, core)
    agent.next_to(subtitle, DOWN, buff=1.0)

    self.play(FadeIn(agent, scale=0.5))
    
    # Uma pulsação calma, mostrando que o agente continua "vivo"
    self.play(Wiggle(core, scale_value=1.2, run_time=2))

    # ==========================================
    # PARTE 3: Abertura para a Banca
    # ==========================================
    
    qa_text = Text("Espaço aberto para perguntas e discussão.", font_size=24, color=LIGHT_GREY)
    qa_text.to_edge(DOWN, buff=1.0)

    self.play(FadeIn(qa_text, shift=UP*0.2))
    
    # Fica em ecrã enquanto a banca faz as perguntas
    self.next_slide()

    # O fade out definitivo quando a defesa terminar
    self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=2)
