from manim import *
import numpy as np

def acoplamento_estrutural(self):
    title = Text("Acoplamento Estrutural", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    p1 = Text("• A Vida é Precária", font_size=28, color=RED_C, weight=BOLD)
    p1_sub = Text(
        "A autopoiese não é garantida.\n"
        "Perturbações podem destruir\n"
        "a organização do sistema.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    
    p2 = Text("• Acoplamento Estrutural", font_size=28, color=GREEN_C, weight=BOLD)
    p2_sub = Text(
        "Interação mútua e contínua.\n"
        "A célula age no ambiente para\n"
        "absorver recursos e sobreviver.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )

    bullets = VGroup(p1, p1_sub, p2, p2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    bullets.to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)

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
    mito1 = Ellipse(width=0.4, height=0.2, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + LEFT * 0.7 + UP * 0.3).rotate(PI/4)
    mito2 = Ellipse(width=0.3, height=0.15, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + RIGHT * 0.5 + DOWN * 0.7).rotate(-PI/6)
    vac1 = Circle(radius=0.15, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + LEFT * 0.4 + DOWN * 0.6)
    vac2 = Circle(radius=0.1, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + RIGHT * 0.8 + UP * 0.6)
    golgi_arcs = VGroup(
        Arc(radius=0.2, angle=PI/1.5, color=TEAL, stroke_width=4),
        Arc(radius=0.3, angle=PI/1.5, color=TEAL, stroke_width=4)
    ).move_to(cell_center + LEFT * 0.2 + UP * 0.8).rotate(PI/4)
    
    organelles = VGroup(mito1, mito2, vac1, vac2, golgi_arcs)
    
    cycle_group.set_opacity(0.3)
    full_cell = VGroup(cytoplasm, membrane, nucleus, organelles, cycle_group)

    cell_label = Text("Sistema Autopoiético", font_size=20, color=YELLOW).next_to(membrane, DOWN, buff=0.7)
    cell_group = VGroup(full_cell, cell_label)

    self.play(FadeIn(cell_group))
    self.next_slide()
    
    self.play(FadeIn(p1, shift=LEFT * 0.2), FadeIn(p1_sub, shift=LEFT * 0.2))

    threat1 = Arrow(start=cell_center + UP*2.0 + RIGHT*2.0, end=cell_center + UP*1.2 + RIGHT*1.2, color=RED, stroke_width=6)
    lbl_threat1 = Text("Perturbações", font_size=16, color=RED).next_to(threat1.get_start(), UP + RIGHT, buff=0.1)

    threat2 = Arrow(start=cell_center + RIGHT*2.6, end=cell_center + RIGHT*1.6, color=RED, stroke_width=6)
    lbl_threat2 = Text("Desintegração", font_size=16, color=RED).next_to(threat2.get_start(), RIGHT, buff=0.1)

    threat3 = Arrow(start=cell_center + DOWN*2.0 + RIGHT*2.0, end=cell_center + DOWN*1.2 + RIGHT*1.2, color=RED, stroke_width=6)
    lbl_threat3 = Text("Dissipação", font_size=16, color=RED).next_to(threat3.get_start(), DOWN + RIGHT, buff=0.1)

    threats = VGroup(threat1, threat2, threat3, lbl_threat1, lbl_threat2, lbl_threat3)

    self.play(
        GrowArrow(threat1), FadeIn(lbl_threat1),
        GrowArrow(threat2), FadeIn(lbl_threat2),
        GrowArrow(threat3), FadeIn(lbl_threat3)
    )
    
    membrane.generate_target()
    membrane.target.set_color(RED)
    
    self.play(
        MoveToTarget(membrane),
        full_cell.animate.scale(0.85, about_point=cell_center),
        Wiggle(full_cell, scale_value=1.05, rotation_angle=0.03*TAU),
        run_time=1.5
    )
    
    self.next_slide()
    
    self.play(
        FadeOut(threats),
        FadeIn(p2, shift=LEFT * 0.2), 
        FadeIn(p2_sub, shift=LEFT * 0.2)
    )

    env_label = Text("Ambiente", font_size=24, color=GRAY).move_to(cell_center + UP*2.5 + RIGHT*1.0)
    self.play(Write(env_label))

    action_arrow = Arrow(start=membrane.get_top(), end=env_label.get_bottom() + LEFT*0.3, color=BLUE, buff=0.1)
    action_lbl = Text("Ação", font_size=16, color=BLUE).next_to(action_arrow, LEFT, buff=0.1)
    
    self.play(GrowArrow(action_arrow), FadeIn(action_lbl))
    
    nutrient = Circle(radius=0.2, color=GREEN, fill_color=GREEN, fill_opacity=0.8)
    nutrient.move_to(env_label.get_bottom() + DOWN*0.5 + RIGHT*0.3)
    nutrient_plus = Text("+", font_size=16, color=WHITE, weight=BOLD).move_to(nutrient.get_center())
    nutrient_group = VGroup(nutrient, nutrient_plus)

    feedback_arrow = Arrow(start=nutrient_group.get_bottom(), end=membrane.get_top() + RIGHT*0.3, color=GREEN, buff=0.1)
    feedback_lbl = Text("Recurso", font_size=16, color=GREEN).next_to(feedback_arrow, RIGHT, buff=0.1)

    self.play(FadeIn(nutrient_group, shift=DOWN*0.2))
    self.play(GrowArrow(feedback_arrow), FadeIn(feedback_lbl))
    self.next_slide()

    self.play(
        nutrient_group.animate.move_to(cell_center),
        FadeOut(action_arrow), FadeOut(action_lbl),
        FadeOut(feedback_arrow), FadeOut(feedback_lbl),
        run_time=1.5
    )
    
    membrane.generate_target()
    membrane.target.set_color(YELLOW)
    
    self.play(
        FadeOut(nutrient_group, scale=0.1), 
        MoveToTarget(membrane),
        full_cell.animate.scale(1 / 0.85, about_point=cell_center), 
        run_time=1.5
    )
    
    self.play(
        full_cell.animate(rate_func=there_and_back, run_time=1.5).scale(1.05, about_point=cell_center)
    )
    self.next_slide()
    # A CENA ACABA AQUI. Sem apagar nada, para fundir com o próximo slide.
