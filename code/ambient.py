from manim import *
import numpy as np

def ambient(self):
    # --- Título ---
    title = Text("O Ambiente: Health-Gathering (VizDoom)", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: A Sala e o Agente (Visão Superior Real)
    # ==========================================
    
    # Carregando a imagem real do cenário
    room = ImageMobject("assets/top-down-health-gathering.png")
    room.height = 5.0 
    room.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5) 
    
    room_label = Text("Visão Superior da Sala", font_size=24).next_to(room, UP)
    acid_label = Text("Chão Ácido!", font_size=20, color=RED).move_to(room.get_bottom() + UP * 0.3)

    self.play(FadeIn(room), Write(room_label), Write(acid_label))

    # O Agente
    agent = Triangle(color=BLUE, fill_opacity=1).scale(0.3).move_to(room.get_center())
    agent_tag = Text("Agente", font_size=18, color=BLUE).next_to(agent, DOWN, buff=0.1)
    agent_group = VGroup(agent, agent_tag)

    self.play(FadeIn(agent_group, shift=UP*0.5))
    self.next_slide()

    # ==========================================
    # PARTE 2: A Barra de Vida e a Mecânica
    # ==========================================
    
    # Informações textuais na direita
    info_title = Text("Mecânicas da Tarefa:", font_size=28, color=YELLOW)
    
    bullet_1 = Text("• Vida inicial: 100", font_size=22)
    bullet_2 = Text("• Perde vida continuamente (-8/36 ticks)", font_size=22)
    bullet_3 = Text("• Duração máxima do episódio: 2100 ticks", font_size=22) # Alterado aqui
    
    info_group = VGroup(info_title, bullet_1, bullet_2, bullet_3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    info_group.to_edge(RIGHT, buff=0.5).shift(UP * 1)

    self.play(Write(info_group))
    self.next_slide()

    # Barra de Vida visual
    health_bar_outline = Rectangle(width=4, height=0.4, color=WHITE)
    health_bar_fill = Rectangle(width=3.9, height=0.3, fill_color=GREEN, fill_opacity=1, stroke_width=0)
    health_bar_fill.move_to(health_bar_outline.get_center())
    
    health_group = VGroup(health_bar_outline, health_bar_fill).next_to(info_group, DOWN, buff=1.5)
    health_label = Text("Vida do Agente", font_size=24).next_to(health_group, UP, buff=0.2)

    self.play(FadeIn(health_label), Create(health_bar_outline), FadeIn(health_bar_fill))
    self.next_slide()

    # Animação da vida a cair pelo chão ácido
    health_bar_fill.generate_target()
    health_bar_fill.target.stretch_to_fit_width(1.5)
    health_bar_fill.target.align_to(health_bar_outline, LEFT).shift(RIGHT*0.05)
    health_bar_fill.target.set_color(RED)

    self.play(
        MoveToTarget(health_bar_fill), 
        run_time=3, 
        rate_func=linear
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: O Salvador (Medikits)
    # ==========================================

    target_pos = room.get_center() + LEFT * 1.5 + UP * 0.2
    
    highlight_circle = Circle(radius=0.4, color=YELLOW, stroke_width=4).move_to(target_pos)
    highlight_text = Text("Medikit", font_size=18, color=YELLOW).next_to(highlight_circle, UP, buff=0.1)
    highlight_group = VGroup(highlight_circle, highlight_text)

    medikit_text = Text("• Medikits (+): Curam 25 de Vida", font_size=22, color=GREEN).next_to(info_group, DOWN, buff=0.4)

    self.play(
        Create(highlight_circle),
        Write(highlight_text),
        Write(medikit_text)
    )
    self.next_slide()

    # Agente move-se até ao medikit em destaque
    dy = target_pos[1] - agent.get_center()[1]
    dx = target_pos[0] - agent.get_center()[0]
    rotation_angle = np.arctan2(dy, dx) - np.pi/2
    
    self.play(
        agent_group.animate.rotate(
            angle=rotation_angle, 
            about_point=agent.get_center()
        )
    )
    
    self.play(
        agent_group.animate.move_to(target_pos),
        run_time=1.5
    )

    # Coleta
    health_bar_fill.generate_target()
    health_bar_fill.target.stretch_to_fit_width(2.8)
    health_bar_fill.target.align_to(health_bar_outline, LEFT).shift(RIGHT*0.05)
    health_bar_fill.target.set_color(YELLOW) 

    self.play(
        FadeOut(highlight_group, scale=1.5),
        MoveToTarget(health_bar_fill),
        Flash(agent_group, color=GREEN, flash_radius=0.5)
    )
    self.next_slide()

    # Transição final (Texto "Objetivo Oculto" removido)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
