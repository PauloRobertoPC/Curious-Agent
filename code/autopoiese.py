from manim import *

def autopoiese(self):
    # --- Títulos ---
    title = Text("Autopoiese", font_size=40, weight=BOLD).to_edge(UP)
    subtitle = Text("(Maturana & Varela, 1980)", font_size=24, color=GRAY).next_to(title, DOWN, buff=0.2)
    
    self.play(Write(title), FadeIn(subtitle))
    self.next_slide()

    # Etimologia
    etymology = Text("auto (si mesmo) + poiesis (produção/criação)", font_size=28, color=YELLOW)
    etymology.next_to(subtitle, DOWN, buff=0.4)
    self.play(Write(etymology))
    self.next_slide()

    self.play(
        etymology.animate.scale(0.8).next_to(subtitle, DOWN, buff=0.2)
    )

    # ==========================================
    # PARTE 1: Os Textos (Focados apenas na Autopoiese)
    # ==========================================
    p1 = Text("• Rede de processos que\n  produz a si mesma.", font_size=24, line_spacing=1)
    p2 = Text("• Estabelece uma fronteira e\n  mantém sua identidade.", font_size=24, line_spacing=1)
    p3 = Text("• Exemplo fundamental:\n  A Célula Biológica.", font_size=24, color=GREEN_C, line_spacing=1)

    bullets = VGroup(p1, p2, p3).arrange(DOWN, aligned_edge=LEFT, buff=0.8).to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)

    # ==========================================
    # PARTE 2: A Representação Visual (A Célula)
    # ==========================================
    cell_center = RIGHT * 3.5 + DOWN * 0.8
    
    # Processos
    node1 = Dot(radius=0.12, color=ORANGE).move_to(cell_center + UP * 0.8 + LEFT * 0.5)
    node2 = Dot(radius=0.12, color=TEAL).move_to(cell_center + RIGHT * 0.8)
    node3 = Dot(radius=0.12, color=RED_C).move_to(cell_center + DOWN * 0.8 + LEFT * 0.5)

    arrow1 = CurvedArrow(node1.get_center(), node2.get_center(), angle=TAU/4, color=WHITE)
    arrow2 = CurvedArrow(node2.get_center(), node3.get_center(), angle=TAU/4, color=WHITE)
    arrow3 = CurvedArrow(node3.get_center(), node1.get_center(), angle=TAU/4, color=WHITE)

    cycle_group = VGroup(node1, node2, node3, arrow1, arrow2, arrow3)
    
    self.play(FadeIn(p1, shift=LEFT * 0.2))
    self.play(Create(node1), Create(node2), Create(node3), Create(arrow1), Create(arrow2), Create(arrow3), run_time=2)
    self.next_slide()

    # Identidade
    membrane = Circle(radius=1.7, color=YELLOW, stroke_width=4)
    membrane.move_to(cell_center)
    membrane_lbl = Text("Identidade\n(Fronteira)", font_size=18, color=YELLOW, line_spacing=1).next_to(membrane, DOWN, buff=0.2)

    self.play(FadeIn(p2, shift=LEFT * 0.2))
    self.play(Create(membrane), Write(membrane_lbl))
    self.next_slide()

    # Célula Viva
    self.play(FadeIn(p3, shift=LEFT * 0.2))
    
    cytoplasm = Circle(radius=1.7, color=YELLOW, fill_color=GREEN_E, fill_opacity=0.3, stroke_width=4).move_to(cell_center)
    nucleus = Circle(radius=0.4, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.8).move_to(cell_center + RIGHT*0.2 + UP*0.2)
    nucleus_lbl = Text("Núcleo", font_size=14, color=WHITE).move_to(nucleus.get_center())
    
    mito1 = Ellipse(width=0.4, height=0.2, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + LEFT * 0.7 + UP * 0.3).rotate(PI/4)
    mito2 = Ellipse(width=0.3, height=0.15, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + RIGHT * 0.5 + DOWN * 0.7).rotate(-PI/6)
    vac1 = Circle(radius=0.15, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + LEFT * 0.4 + DOWN * 0.6)
    vac2 = Circle(radius=0.1, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + RIGHT * 0.8 + UP * 0.6)
    golgi_arcs = VGroup(Arc(radius=0.2, angle=PI/1.5, color=TEAL, stroke_width=4), Arc(radius=0.3, angle=PI/1.5, color=TEAL, stroke_width=4)).move_to(cell_center + LEFT * 0.2 + UP * 0.8).rotate(PI/4)
    
    organelles = VGroup(mito1, mito2, vac1, vac2, golgi_arcs)
    cell_label = Text("Célula Viva", font_size=24, color=GREEN_C, weight=BOLD).next_to(membrane, UP, buff=0.2)

    self.play(
        FadeIn(cytoplasm), FadeIn(nucleus), Write(nucleus_lbl), FadeIn(organelles), Write(cell_label),
        cycle_group.animate.set_opacity(0.3) 
    )
    
    full_cell = VGroup(membrane, cytoplasm, nucleus, nucleus_lbl, organelles, cycle_group)
    self.play(full_cell.animate(rate_func=there_and_back, run_time=1.5).scale(1.05, about_point=cell_center))
    self.next_slide()

    # ==========================================
    # PARTE 3: Transição contínua para o próximo slide
    # ==========================================
    
    # Calculando a nova posição desejada no slide de acoplamento[cite: 1]
    new_cell_center = RIGHT * 1.5 + DOWN * 0.5
    shift_vector = new_cell_center - cell_center

    new_cell_label = Text("Sistema Autopoiético", font_size=20, color=YELLOW).next_to(membrane.copy().shift(shift_vector), DOWN, buff=0.7)

    # Movemos a célula inteira e apagamos apenas os textos
    self.play(
        FadeOut(title), FadeOut(subtitle), FadeOut(etymology), 
        FadeOut(bullets), FadeOut(membrane_lbl),
        full_cell.animate.shift(shift_vector),
        Transform(cell_label, new_cell_label),
        run_time=1.5
    )
    self.next_slide()
    # FIM: A célula não sofre FadeOut. Ela será absorvida pelo próximo slide.
