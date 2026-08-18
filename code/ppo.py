from manim import *
import numpy as np

def ppo(self):
    # --- Título Inicial ---
    title = Text("Policy Gradient", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: A Derivação (Simples)
    # ==========================================
    
    # 1. Trajetória
    traj_text = Text("1. Trajetória (", font_size=28)
    traj_math = MathTex(r"\tau", font_size=32, color=YELLOW)
    traj_desc = Text("): Sequência de estados e ações", font_size=28)
    traj_group = VGroup(traj_text, traj_math, traj_desc).arrange(RIGHT, buff=0.1)

    # 2. Objetivo
    obj_text = Text("2. Objetivo:", font_size=28)
    obj_desc = Text("Maximizar o Retorno Esperado", font_size=28, color=BLUE)
    obj_math = MathTex(r"J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]", font_size=38)
    obj_group = VGroup(
        VGroup(obj_text, obj_desc).arrange(RIGHT, buff=0.2), 
        obj_math
    ).arrange(DOWN, buff=0.3)

    # 3. Gradiente
    grad_text = Text("3. Gradiente:", font_size=28)
    grad_desc = Text("Ajustar pesos nas direções de maior recompensa", font_size=28, color=GREEN)
    grad_math = MathTex(r"\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau) \right]", font_size=38)
    grad_group = VGroup(
        VGroup(grad_text, grad_desc).arrange(RIGHT, buff=0.2), 
        grad_math
    ).arrange(DOWN, buff=0.3)

    # Agrupando e posicionando
    math_section = VGroup(traj_group, obj_group, grad_group).arrange(DOWN, buff=0.6).next_to(title, DOWN, buff=0.5)

    if math_section.height > 5.5:
        math_section.scale(5.5 / math_section.height)

    self.play(FadeIn(traj_group, shift=UP * 0.2))
    self.next_slide()
    self.play(FadeIn(obj_group, shift=UP * 0.2))
    self.next_slide()
    self.play(FadeIn(grad_group, shift=UP * 0.2))
    self.next_slide()

    self.play(FadeOut(math_section))


    # ==========================================
    # PARTE 2: Analogia Visual (O Passo Imprudente)
    # ==========================================

    # Mudando o título suavemente
    new_title = Text("O Problema do Policy Gradient", font_size=40, weight=BOLD).to_edge(UP)
    subtitle = Text("O Risco: Passos Muito Grandes", font_size=32, color=RED).next_to(new_title, DOWN, buff=0.2)
    
    self.play(
        Transform(title, new_title),
        Write(subtitle)
    )

    ax = Axes(
        x_range=[0, 8, 1],
        y_range=[0, 6, 1],
        x_length=10,
        y_length=4.5,
        axis_config={"color": BLACK} 
    ).shift(DOWN * 0.5)

    def hill_func(x):
        return 3.5 * np.exp(-0.5 * (x - 3)**2) - 2 * np.exp(-1.5 * (x - 5.5)**2) + 2.5

    mountain = ax.plot(hill_func, x_range=[0.5, 7.5], color=WHITE, stroke_width=4)
    mountain_fill = ax.get_area(mountain, x_range=[0.5, 7.5], color=[DARK_GREY, BLACK], opacity=0.5)
    mountain_group = VGroup(mountain_fill, mountain)
    
    self.play(Create(mountain_group))
    
    agent_x = ValueTracker(1.2) 
    agent = always_redraw(
        lambda: Dot(ax.c2p(agent_x.get_value(), hill_func(agent_x.get_value())), color=BLUE, radius=0.15)
    )
    
    self.play(FadeIn(agent, scale=0.5))
    self.next_slide()

    x_val = 1.2
    dx = 0.01
    dy = hill_func(x_val + dx) - hill_func(x_val)
    angle = np.arctan2(dy, dx)
    
    # Seta ligeiramente menor para não bater no título
    arrow_length = 3.0
    start_pt = ax.c2p(x_val, hill_func(x_val))
    end_pt = ax.c2p(x_val + arrow_length * np.cos(angle), hill_func(x_val) + arrow_length * np.sin(angle))
    
    grad_arrow = Arrow(start_pt, end_pt, color=YELLOW, buff=0, stroke_width=6)
    # Posicionado à direita em vez de acima para evitar o subtítulo
    grad_label = Text("Sinal do Gradiente", font_size=24, color=YELLOW).next_to(grad_arrow, RIGHT, buff=0.1)

    self.play(GrowArrow(grad_arrow), FadeIn(grad_label))
    self.next_slide()

    intended_step = DashedLine(start_pt, end_pt, color=YELLOW, stroke_width=2)
    
    self.play(
        FadeOut(grad_arrow),
        FadeOut(grad_label),
        FadeIn(intended_step),
        agent_x.animate.set_value(5.8), 
        run_time=2.5,
        rate_func=rush_from 
    )
    
    disaster_text = Text("Colapso da Política!", font_size=32, color=RED).next_to(agent, UP, buff=0.5)
    
    self.play(
        agent.animate.set_color(RED),
        Write(disaster_text),
        Flash(agent, color=RED, line_length=0.4, flash_radius=0.4) 
    )
    self.next_slide()


    # ==========================================
    # PARTE 3: A Solução (Região de Confiança / PPO)
    # ==========================================

    # Rebobinando a cena (Apaga o desastre e volta o agente)
    self.play(
        FadeOut(disaster_text),
        FadeOut(intended_step),
        agent.animate.set_color(BLUE),
        agent_x.animate.set_value(1.2),
        run_time=1.5
    )

    # Muda o título para a solução
    ppo_title = Text("Proximal Policy Optimization (PPO)", font_size=40, weight=BOLD).to_edge(UP)
    ppo_subtitle = Text("A Solução: Região de Confiança", font_size=32, color=GREEN).next_to(ppo_title, DOWN, buff=0.2)
    
    self.play(
        Transform(title, ppo_title),
        Transform(subtitle, ppo_subtitle)
    )
    self.next_slide()

    # Calculando a posição do passo seguro (limitado a x=2.5 por exemplo)
    safe_x_val = 2.5
    safe_end_pt = ax.c2p(safe_x_val, hill_func(safe_x_val))
    
    # O raio do círculo é exatamente a distância visual entre o agente e o ponto seguro
    circle_radius = np.linalg.norm(safe_end_pt - start_pt)

    # Desenhando o Círculo da Região de Confiança
    trust_region = Circle(radius=circle_radius, color=GREEN, fill_opacity=0.15, stroke_width=3).move_to(start_pt)
    trust_label = Text("Trust Region", font_size=20, color=GREEN).next_to(trust_region, DOWN, buff=0.1)
    
    self.play(Create(trust_region), FadeIn(trust_label))
    self.next_slide()

    # O Algoritmo tenta dar um passo enorme novamente...
    grad_arrow_2 = Arrow(start_pt, end_pt, color=YELLOW, buff=0, stroke_width=6)
    self.play(GrowArrow(grad_arrow_2))
    
    # ...Mas a Região de Confiança "corta" (Clip) a seta na borda do círculo
    clipped_arrow = Arrow(start_pt, safe_end_pt, color=GREEN, buff=0, stroke_width=6)
    clipping_text = Text("Clipping!", font_size=24, color=GREEN).next_to(clipped_arrow, UP, buff=0.1)
    
    self.play(
        Transform(grad_arrow_2, clipped_arrow),
        FadeIn(clipping_text)
    )
    self.next_slide()

    # O agente dá o passo de forma segura sem cair do penhasco
    safe_step = DashedLine(start_pt, safe_end_pt, color=GREEN, stroke_width=2)
    
    self.play(
        FadeOut(grad_arrow_2),
        FadeOut(clipping_text),
        FadeIn(safe_step),
        agent_x.animate.set_value(safe_x_val),
        run_time=2
    )

    success_text = Text("Passo Seguro!", font_size=32, color=GREEN).next_to(agent, DOWN, buff=0.5)
    self.play(Write(success_text))
    self.next_slide()
    self.clear()
