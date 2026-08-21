from manim import *

def ia_enativa(self):
    # Limpa a tela para a introdução do novo tópico
    self.remove(*self.mobjects)

    # --- Título ---
    title = Text("IA Enativa", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: Princípios de Design (Froese & Ziemke)
    # ==========================================
    
    lbl_design = Text("Princípios de Design:", font_size=28, color=GRAY)
    
    # Princípio 1
    p1 = Text("1. Autonomia Constitutiva", font_size=28, color=BLUE_C, weight=BOLD)
    p1_sub = Text(
        "A capacidade do sistema (autopoiese) de gerar\n"
        "e manter a sua própria identidade.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    group_p1 = VGroup(p1, p1_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    
    # Princípio 2
    p2 = Text("2. Adaptatividade", font_size=28, color=ORANGE, weight=BOLD)
    p2_sub = Text(
        "A capacidade de monitorizar a sua viabilidade\n"
        "e regular a interação com o ambiente.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    group_p2 = VGroup(p2, p2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    
    # Agrupamento à esquerda
    principles = VGroup(lbl_design, group_p1, group_p2).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
    principles.to_edge(LEFT, buff=0.8).shift(UP * 0.4)

    self.play(FadeIn(lbl_design))
    self.play(FadeIn(group_p1, shift=LEFT*0.2))
    self.next_slide()
    
    self.play(FadeIn(group_p2, shift=LEFT*0.2))
    self.next_slide()

    # A Síntese (Sense-making)
    p3 = Text("Autonomia + Adaptatividade = Produção de Sentido", font_size=24, color=GREEN_C, weight=BOLD)
    p3_sub = Text("(FROESE; ZIEMKE, 2009)", font_size=20, color=YELLOW)
    
    group_p3 = VGroup(p3, p3_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    group_p3.next_to(principles, DOWN, buff=0.8, aligned_edge=LEFT)

    self.play(FadeIn(group_p3, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: A Célula (Perda e Ganho de Viabilidade)
    # ==========================================
    
    # Posicionando a célula no lado direito
    cell_center = RIGHT * 3.5 + DOWN * 0.5
    
    membrane = Circle(radius=1.2, color=YELLOW, stroke_width=4).move_to(cell_center)
    cytoplasm = Circle(radius=1.2, color=YELLOW, fill_color=GREEN_E, fill_opacity=0.3, stroke_width=0).move_to(cell_center)
    nucleus = Circle(radius=0.3, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.8).move_to(cell_center)
    cell = VGroup(cytoplasm, membrane, nucleus)
    
    # Barra de Viabilidade
    bar_bg = Rectangle(width=2.0, height=0.2, color=WHITE).next_to(membrane, DOWN, buff=0.5)
    bar_fill = Rectangle(width=1.9, height=0.15, color=GREEN, fill_color=GREEN, fill_opacity=0.8)
    bar_fill.move_to(bar_bg.get_left() + RIGHT*0.05, aligned_edge=LEFT)
    lbl_bar = Text("Viabilidade", font_size=16).next_to(bar_bg, DOWN, buff=0.1)
    
    health_bar = VGroup(bar_bg, bar_fill, lbl_bar)
    agent = VGroup(cell, health_bar)
    
    self.play(FadeIn(agent))
    self.next_slide()

    # Animação: A célula perde vida (Precariedade em ação)
    bar_fill.generate_target()
    bar_fill.target.stretch_to_fit_width(0.3, about_edge=LEFT) # Barra desce drasticamente
    bar_fill.target.set_color(RED)
    
    self.play(
        MoveToTarget(bar_fill),
        cytoplasm.animate.set_fill(RED_E, opacity=0.5), # O interior fica vermelho escuro
        membrane.animate.set_color(RED),                # A membrana fica vermelha
        nucleus.animate.set_color(RED_A),               # O núcleo sofre stress
        run_time=2
    )
    
    # A célula "treme" simbolizando o risco de dissipação
    self.play(Wiggle(cell, scale_value=1.05, rotation_angle=0.03*TAU, run_time=1.5))
    self.next_slide()

    # Surge o Recurso no ambiente
    resource = RegularPolygon(n=6, color=GREEN, fill_color=GREEN, fill_opacity=0.8).scale(0.3)
    resource.move_to(cell_center + UP * 2.5 + LEFT * 1.5)
    plus = Text("+", font_size=20, weight=BOLD).move_to(resource)
    res_group = VGroup(resource, plus)
    
    self.play(FadeIn(res_group, shift=DOWN*0.2))
    
    # A Célula adquire o recurso (Ação via adaptatividade)
    self.play(
        res_group.animate.move_to(nucleus.get_center()),
        run_time=1.5
    )
    
    # Animação: A célula recupera a vida
    bar_fill.generate_target()
    bar_fill.target.stretch_to_fit_width(1.9, about_edge=LEFT) # Barra volta ao topo
    bar_fill.target.set_color(GREEN)
    
    self.play(
        FadeOut(res_group, scale=0.1), # O recurso é "digerido"
        MoveToTarget(bar_fill),
        cytoplasm.animate.set_fill(GREEN_E, opacity=0.3), # Volta ao estado saudável
        membrane.animate.set_color(YELLOW),
        nucleus.animate.set_color(BLUE_E),
        run_time=1.5
    )
    
    # Brilho de estabilidade
    self.play(Flash(membrane, color=GREEN, line_length=0.3, num_lines=12))
    self.next_slide()

    # ==========================================
    # PARTE 3: A Grande Conclusão
    # ==========================================
    
    # Limpamos o diagrama e a teoria para focar 100% na frase final
    self.play(
        FadeOut(principles), 
        FadeOut(group_p3), 
        FadeOut(agent)
    )
    
    # O Clímax textual
    quote = Text(
        "Sob a ótica da IA Enativa, a inteligência\n"
        "pode ser reinterpretada como a capacidade de um\n"
        "sistema de lidar com sua própria precariedade.", 
        font_size=36, 
        weight=BOLD, 
        line_spacing=1.5, 
        # Dá cor às palavras-chave para ancorar visualmente o conceito
        t2c={"inteligência": BLUE_C, "própria precariedade": RED_C} 
    )
    quote.move_to(ORIGIN)
    
    self.play(Write(quote, run_time=3.0))
    self.next_slide()
    
    # Limpa a tela para o próximo slide
    self.play(*[FadeOut(mob) for mob in self.mobjects])
