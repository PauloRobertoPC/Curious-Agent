from manim import *

def ponte_pratica(self):
    # Limpa a tela para a introdução do novo tópico
    self.remove(*self.mobjects)

    # --- Título ---
    title = Text("Da Teoria à Prática", font_size=40, weight=BOLD).to_edge(UP)
    subtitle = Text("(O Desafio da Implementação)", font_size=24, color=GRAY).next_to(title, DOWN, buff=0.2)
    
    self.play(Write(title), FadeIn(subtitle, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 1: A Tríade da IA Enativa (Qualidades da Vida)
    # ==========================================
    
    # Criamos três caixas para mostrar como uma coisa leva à outra
    box_prec = RoundedRectangle(width=3.5, height=1.5, color=RED_C, fill_color=RED, fill_opacity=0.2)
    text_prec = Text("Precariedade\n(Risco de Morte)", font_size=24, color=WHITE, line_spacing=1).move_to(box_prec)
    node_prec = VGroup(box_prec, text_prec)

    box_auto = RoundedRectangle(width=3.5, height=1.5, color=BLUE_C, fill_color=BLUE, fill_opacity=0.2)
    text_auto = Text("Autonomia\n(Intencionalidade)", font_size=24, color=WHITE, line_spacing=1).move_to(box_auto)
    node_auto = VGroup(box_auto, text_auto)

    box_sense = RoundedRectangle(width=3.5, height=1.5, color=GREEN_C, fill_color=GREEN, fill_opacity=0.2)
    text_sense = Text("Produção de Sentido\n(Compreensão)", font_size=24, color=WHITE, line_spacing=1).move_to(box_sense)
    node_sense = VGroup(box_sense, text_sense)

    # Posiciona os nós em linha horizontal
    flow = VGroup(node_prec, node_auto, node_sense).arrange(RIGHT, buff=1.0).move_to(UP * 0.5)
    
    # Setas conectando os nós
    arrow1 = Arrow(start=node_prec.get_right(), end=node_auto.get_left(), color=WHITE, buff=0.1)
    arrow2 = Arrow(start=node_auto.get_right(), end=node_sense.get_left(), color=WHITE, buff=0.1)

    # Animação do fluxo lógico da IA Enativa
    self.play(FadeIn(node_prec, shift=RIGHT*0.2))
    self.play(GrowArrow(arrow1))
    self.play(FadeIn(node_auto, shift=RIGHT*0.2))
    self.play(GrowArrow(arrow2))
    self.play(FadeIn(node_sense, shift=RIGHT*0.2))
    
    # Destaca que isso é o que falta na IA tradicional
    brace = Brace(flow, DOWN, color=YELLOW)
    brace_text = brace.get_text("As qualidades essenciais de um ser vivo").set_color(YELLOW)
    
    self.play(GrowFromCenter(brace), FadeIn(brace_text, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: A Mudança de Perspectiva (O Problema de Engenharia)
    # ==========================================
    
    # Apaga o fluxo para focar nas perguntas cruciais
    self.play(
        FadeOut(flow), 
        FadeOut(arrow1), FadeOut(arrow2),
        FadeOut(brace), FadeOut(brace_text)
    )

    # Introdução às perguntas
    challenge_lbl = Text("Como simular isso num agente de Aprendizado por Reforço?", font_size=32, color=BLUE_C, weight=BOLD).shift(UP * 1.5)
    self.play(Write(challenge_lbl))
    self.next_slide()

    # Pergunta 1: O Corpo
    q1_num = Text("1.", font_size=36, color=YELLOW, weight=BOLD)
    q1_text = Text(
        "Como podemos tornar o corpo do\n"
        "agente precário?", 
        font_size=32, line_spacing=1
    )
    q1 = VGroup(q1_num, q1_text).arrange(RIGHT, aligned_edge=UP, buff=0.3)

    # Pergunta 2: A Recompensa
    q2_num = Text("2.", font_size=36, color=YELLOW, weight=BOLD)
    q2_text = Text(
        "Como transformar a luta contra a\n"
        "precariedade num sinal de recompensa?", 
        font_size=32, line_spacing=1
    )
    q2 = VGroup(q2_num, q2_text).arrange(RIGHT, aligned_edge=UP, buff=0.3)

    # Agrupa e alinha as perguntas no centro
    questions = VGroup(q1, q2).arrange(DOWN, aligned_edge=LEFT, buff=0.8).next_to(challenge_lbl, DOWN, buff=1.0)
    
    self.play(FadeIn(q1, shift=LEFT*0.2))
    self.next_slide()
    
    self.play(FadeIn(q2, shift=LEFT*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 3: O Gancho para o Próximo Slide (O Corpo / Glaucoma)
    # ==========================================
    
    # Destaca especificamente a Pergunta 1, que será respondida a seguir com o glaucoma
    box_body = SurroundingRectangle(q1_text, color=RED_C, buff=0.2)
    self.play(Create(box_body))
    self.next_slide()

    # Limpa a tela para o slide seguinte
    self.play(*[FadeOut(mob) for mob in self.mobjects])
