from manim import *

def deep_reinforcement_learning(self):
    # --- Title ---
    title = Text("Aprendizado por Reforço Profundo", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # --- PART 1: The Curse of Dimensionality ---
    # Grouping the state elements so they move together
    state_text = Text("Estado Contínuo\n(Pixels da Imagem):", font_size=24, line_spacing=1)
    state_array = MathTex(
        r"S_t = \begin{bmatrix} 255 & 128 & \dots \\ 45 & 200 & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}"
    ).scale(0.8)
    state_size = Text("Ex: 84x84 = 7.056 variáveis", font_size=18, color=YELLOW)

    # Arrange vertically and pin firmly to the left edge
    state_group = VGroup(state_text, state_array, state_size).arrange(DOWN, buff=0.4).to_edge(LEFT, buff=0.5).shift(DOWN * 0.2)

    self.play(Write(state_group[0]), FadeIn(state_group[1]), FadeIn(state_group[2]))
    self.next_slide()

    # Traditional approach: Q-Table
    table_text = Text("Tabela Q (Tradicional)", font_size=24)
    q_table = MathTex(
        r"\begin{bmatrix} Q(s_1, a_1) & \dots \\ \vdots & \ddots \end{bmatrix}"
    )
    # Group and pin to the right
    q_group = VGroup(table_text, q_table).arrange(DOWN, buff=0.4).to_edge(RIGHT, buff=1.5).shift(DOWN * 0.2)
    
    self.play(Write(q_group[0]), FadeIn(q_group[1]))
    self.next_slide()

    # The problem: Millions of pixel combinations crash the table
    curse_text = Text(
        "Dimensão massiva!\nImpossível mapear tudo.", 
        font_size=20, 
        color=RED
    ).next_to(q_group, DOWN, buff=0.5)
    cross = Cross(q_group[1])

    self.play(Create(cross), Write(curse_text))
    self.next_slide()

    # --- PART 2: The Zoom-In (Convolutional Neural Network) ---
    # Clear the failing Q-Table
    self.play(
        FadeOut(q_group), FadeOut(cross), FadeOut(curse_text)
    )

    # Bring in the "Agente" box, slightly smaller to fit the action matrix later
    agent_box = Rectangle(width=5.8, height=4.2, color=BLUE)
    agent_label = Text("Agente (Rede Neural Convolucional)", font_size=22).next_to(agent_box, UP, buff=0.2)
    
    # Position the agent box dynamically relative to the state group
    agent_ui = VGroup(agent_box, agent_label).next_to(state_group, RIGHT, buff=0.7).shift(DOWN * 0.2)

    self.play(Create(agent_box), Write(agent_label))
    
    # Build a CNN representation INSIDE the box using box coordinates
    input_grid = VGroup(*[Square(side_length=0.2, color=WHITE, fill_opacity=0.2) for _ in range(9)])
    input_grid.arrange_in_grid(rows=3, cols=3, buff=0.05).move_to(agent_box.get_left() + RIGHT * 1.0)
    
    hidden_layer = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.2) for _ in range(5)])
    hidden_layer.arrange(DOWN, buff=0.3).move_to(agent_box.get_center())
    
    output_layer = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.2) for _ in range(2)])
    output_layer.arrange(DOWN, buff=0.3).move_to(agent_box.get_right() + LEFT * 1.0)

    # Draw connecting lines (Now connecting each grid square to mimic a true dense flattening)
    connections = VGroup()
    for square in input_grid:
        for h_node in hidden_layer:
            connections.add(Line(square.get_right(), h_node.get_left(), stroke_width=1, stroke_opacity=0.15))
            
    for h_node in hidden_layer:
        for o_node in output_layer:
            connections.add(Line(h_node.get_right(), o_node.get_left(), stroke_width=1, stroke_opacity=0.3))

    self.play(
        Create(input_grid), 
        Create(hidden_layer), 
        Create(output_layer)
    )
    self.play(Create(connections))
    self.next_slide()

    # --- PART 3: Connecting the Flow ---
    # Connect the image state to the CNN input
    state_arrow = Arrow(state_group[1].get_right(), input_grid.get_left(), color=WHITE, buff=0.2)
    
    # Define the output (Action) as motor torques
    action_text = Text("Ação (Motores):", font_size=24)
    action_array = MathTex(
        r"A_t = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \\ a_4 \end{bmatrix}"
    ).scale(0.8)
    
    # Group and position relative to the agent box
    action_group = VGroup(action_text, action_array).arrange(DOWN, buff=0.3).next_to(agent_box, RIGHT, buff=0.5)
    
    action_arrow = Arrow(output_layer.get_right(), action_group[1].get_left(), color=YELLOW, buff=0.2)

    self.play(Create(state_arrow))
    self.play(Write(action_group[0]), FadeIn(action_group[1]), Create(action_arrow))
    self.next_slide()

    # Clear screen for the next slide
    self.play(*[FadeOut(mob) for mob in self.mobjects])
