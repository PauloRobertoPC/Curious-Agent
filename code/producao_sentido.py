from manim import *
import numpy as np

def producao_sentido(self):
    self.remove(*self.mobjects)

    # ==========================================
    # PARTE 1: Continuidade Visual (A Célula herdada)
    # ==========================================
    cell_center = RIGHT * 1.5 + DOWN * 0.5
    
    node1 = Dot(radius=0.12, color=ORANGE).move_to(cell_center + UP * 0.8 + LEFT * 0.5)
    node2 = Dot(radius=0.12, color=TEAL).move_to(cell_center + RIGHT * 0.8)
    node3 = Dot(radius=0.12, color=RED_C).move_to(cell_center + DOWN * 0.8 + LEFT * 0.5)

    arrow1 = CurvedArrow(node1.get_center(), node2.get_center(), angle=TAU/4, color=WHITE)
    arrow2 = CurvedArrow(node2.get_center(), node3.get_center(), angle=TAU/4, color=WHITE)
    arrow3 = CurvedArrow(node3.get_center(), node1.get_center(), angle=TAU/4, color=WHITE)
    cycle_group = VGroup(node1, node2, node3, arrow1, arrow2, arrow3)
    
    membrane = Circle(radius=1.7, color=YELLOW, stroke_width=4).move_to(cell_center)
    cytoplasm = Circle(radius=1.7, color=YELLOW, fill_color=GREEN_E, fill_opacity=0.3, stroke_width=0).move_to(cell_center)
    
    nucleus = Circle(radius=0.4, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.8).move_to(cell_center + RIGHT*0.2 + UP*0.2)
    nucleus_lbl = Text("Núcleo", font_size=14, color=WHITE).move_to(nucleus.get_center())
    
    mito1 = Ellipse(width=0.4, height=0.2, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + LEFT * 0.7 + UP * 0.3).rotate(PI/4)
    mito2 = Ellipse(width=0.3, height=0.15, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + RIGHT * 0.5 + DOWN * 0.7).rotate(-PI/6)
    vac1 = Circle(radius=0.15, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + LEFT * 0.4 + DOWN * 0.6)
    vac2 = Circle(radius=0.1, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + RIGHT * 0.8 + UP * 0.6)
    golgi_arcs = VGroup(Arc(radius=0.2, angle=PI/1.5, color=TEAL, stroke_width=4), Arc(radius=0.3, angle=PI/1.5, color=TEAL, stroke_width=4)).move_to(cell_center + LEFT * 0.2 + UP * 0.8).rotate(PI/4)
    
    organelles = VGroup(mito1, mito2, vac1, vac2, golgi_arcs)
    
    cycle_group.set_opacity(0.3)
    full_cell = VGroup(cytoplasm, membrane, nucleus, nucleus_lbl, organelles, cycle_group)

    cell_label = Text("Sistema Autopoiético", font_size=20, color=YELLOW).next_to(membrane, DOWN, buff=0.7)
    cell_group = VGroup(full_cell, cell_label)

    # Textos do slide anterior para garantir a transição invisível
    old_title = Text("Acoplamento Estrutural", font_size=40, weight=BOLD).to_edge(UP)
    p1 = Text("• A Vida é Precária", font_size=28, color=RED_C, weight=BOLD)
    p1_sub = Text("A autopoiese não é garantida.\nPerturbações podem destruir\na organização do sistema.", font_size=22, color=LIGHT_GREY, line_spacing=1)
    p2 = Text("• Acoplamento Estrutural", font_size=28, color=GREEN_C, weight=BOLD)
    p2_sub = Text("Interação mútua e contínua.\nA célula age no ambiente para\nabsorver recursos e sobreviver.", font_size=22, color=LIGHT_GREY, line_spacing=1)
    
    old_bullets = VGroup(p1, p1_sub, p2, p2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)
    old_env_label = Text("Ambiente", font_size=24, color=GRAY).move_to(cell_center + UP*2.5 + RIGHT*1.0)

    self.add(old_title, old_bullets, old_env_label, cell_group)

    # ==========================================
    # PARTE 2: O Cross-Fade de Cinema
    # ==========================================
    title = Text("Produção de Sentido", font_size=40, weight=BOLD).to_edge(UP)
    subtitle = Text("(Sense-Making)", font_size=24, color=YELLOW).next_to(title, DOWN, buff=0.2)
    
    self.play(
        FadeOut(old_title), FadeOut(old_bullets), FadeOut(old_env_label),
        Write(title), FadeIn(subtitle, shift=UP*0.2)
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: Os Novos Textos (Foco na Identidade)
    # ==========================================
    t1 = Text("• O mundo não é neutro", font_size=28, color=YELLOW, weight=BOLD)
    t1_sub = Text(
        "A célula avalia o ambiente com base\n"
        "no que contribui ou não para a\n"
        "manutenção da sua própria identidade.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    
    group_t1 = VGroup(t1, t1_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.8).shift(UP * 0.5)
    
    self.play(FadeIn(t1, shift=LEFT*0.2), FadeIn(t1_sub, shift=LEFT*0.2))

    # Objetos no ambiente a serem avaliados
    obj_top = Circle(radius=0.4, color=GRAY, fill_color=GRAY, fill_opacity=0.5).move_to(RIGHT * 5.0 + UP * 1.5)
    obj_bottom = Triangle(color=GRAY, fill_color=GRAY, fill_opacity=0.5).scale(0.5).move_to(RIGHT * 5.0 + DOWN * 1.0)
    
    q_mark_top = Text("?", font_size=24, color=WHITE).move_to(obj_top)
    q_mark_bottom = Text("?", font_size=24, color=WHITE).move_to(obj_bottom)

    neutral_top = VGroup(obj_top, q_mark_top)
    neutral_bottom = VGroup(obj_bottom, q_mark_bottom)

    self.play(FadeIn(neutral_top, shift=LEFT*0.5), FadeIn(neutral_bottom, shift=LEFT*0.5))
    self.next_slide()

    # ==========================================
    # PARTE 4: A Avaliação pela Autopoiese
    # ==========================================
    line_top = DashedLine(start=membrane.get_right(), end=obj_top.get_left(), color=YELLOW, dash_length=0.1)
    line_bottom = DashedLine(start=membrane.get_right(), end=obj_bottom.get_left(), color=YELLOW, dash_length=0.1)

    self.play(Create(line_top), Create(line_bottom))
    self.play(full_cell.animate(rate_func=there_and_back, run_time=1.5).scale(1.05, about_point=cell_center))
    self.next_slide()

    # ==========================================
    # PARTE 5: A Emergência do Significado (Reescrito sem +1/-1)
    # ==========================================
    t2 = Text("• O Significado Emerge", font_size=28, color=GREEN_C, weight=BOLD)
    t2_sub = Text(
        "O que favorece a identidade\n"
        "torna-se um recurso (atração).\n"
        "O que a ameaça torna-se\n"
        "um perigo (repulsa).", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    
    group_t2 = VGroup(t2, t2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(group_t1, DOWN, buff=0.6, aligned_edge=LEFT)
    self.play(FadeIn(group_t2, shift=LEFT*0.2))

    good_obj = Circle(radius=0.4, color=GREEN, fill_color=GREEN, fill_opacity=0.8).move_to(obj_top)
    plus_sign = Text("+", font_size=32, color=WHITE, weight=BOLD).move_to(good_obj)
    good_group = VGroup(good_obj, plus_sign)

    bad_obj = Triangle(color=RED, fill_color=RED, fill_opacity=0.8).scale(0.5).move_to(obj_bottom)
    minus_sign = Text("-", font_size=32, color=WHITE, weight=BOLD).move_to(bad_obj)
    bad_group = VGroup(bad_obj, minus_sign)

    self.play(
        Transform(neutral_top, good_group),
        line_top.animate.set_color(GREEN),
        Transform(neutral_bottom, bad_group),
        line_bottom.animate.set_color(RED),
        run_time=1.5
    )
    
    self.play(
        Flash(good_obj, color=GREEN, line_length=0.2, num_lines=8),
        Flash(bad_obj, color=RED, line_length=0.2, num_lines=8)
    )
    self.next_slide()

    # ==========================================
    # PARTE 6: A Ação (E Fim de cena INTACTO)
    # ==========================================
    self.play(
        FadeOut(line_top), FadeOut(line_bottom),
        cell_group.animate.shift(UP * 0.5 + RIGHT * 0.5),
        run_time=1.5
    )
    self.next_slide()
    # A cena TERMINA AQUI. Com os textos, título e a célula na tela.
    # O mundo_ambiente.py fará a transição invisível apagando o que for necessário.
