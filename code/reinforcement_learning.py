from manim import *

def reinforcement_learning(self):
    # 1. First the box where is written "agente" appears
    agent_box = Rectangle(width=4, height=2, color=BLUE)
    agent_label = Text("Agente")
    agent = VGroup(agent_box, agent_label).shift(UP * 1.5)

    self.play(Create(agent_box), Write(agent_label))
    self.next_slide()

    # 2. Then the box where is written "ambiente" appears
    env_box = Rectangle(width=4, height=2, color=GREEN)
    env_label = Text("Ambiente")
    env = VGroup(env_box, env_label).shift(DOWN * 1.5)

    self.play(Create(env_box), Write(env_label))
    self.next_slide()

    # 3. Then the arrow of the action appears
    # Arrow on the right side pointing from Agente to Ambiente
    action_arrow = CurvedArrow(
        agent.get_right(), 
        env.get_right(), 
        angle=-TAU/4, 
        color=YELLOW
    )
    action_label = Text("Ação", font_size=28).next_to(action_arrow, RIGHT)

    self.play(Create(action_arrow), Write(action_label))
    self.next_slide()

    # 4. Then the others arrows appear
    # Arrow on the left side for State
    state_arrow = CurvedArrow(
        env.get_left(), 
        agent.get_left(), 
        angle=-TAU/4, 
        color=WHITE
    )
    state_label = Text("Estado", font_size=28).next_to(state_arrow, LEFT)

    # Arrow crossing the middle for Reward
    reward_arrow = Line(
        env.get_top(), 
        agent.get_bottom(), 
        color=RED
    ).add_tip()
    reward_label = Text("Recompensa", font_size=28).next_to(reward_arrow, RIGHT)

    self.play(
        Create(state_arrow), Write(state_label),
        Create(reward_arrow), Write(reward_label)
    )
    self.next_slide()

    # 5. Make something flow in the arrow while the "Retorno" eq is being built
    retorno_eq = MathTex(
        r"G_t = \gamma^{0} R_{t} + \gamma^{1} R_{t+1} + \gamma^{2} R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}"
    ).scale(0.8).to_edge(UP, buff=0.1)

    # The object that will "flow" along the reward arrow
    flow_dot = Dot(color=YELLOW, radius=0.1)

    self.play(
        # Write the equation
        Write(retorno_eq, run_time=2.5),
        # Animate the dot moving from the start to the end of the reward arrow
        MoveAlongPath(flow_dot, reward_arrow, run_time=2.5, rate_func=linear)
    )
    self.play(FadeOut(flow_dot))

    self.next_slide()
    self.clear()
