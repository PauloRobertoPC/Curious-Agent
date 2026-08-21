from manim import *

def adaptatividade(self):
    # Limpa o ecrã do slide anterior
    self.remove(*self.mobjects)

    # --- Título ---
    title = Text("Adaptatividade", font_size=40, weight=BOLD).to_edge(UP)
    subtitle = Text("(Ezequiel Di Paolo, 2005)", font_size=24, color=YELLOW).next_to(title, DOWN, buff=0.2)
    
    self.play(Write(title), FadeIn(subtitle, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 1: Os Textos à Esquerda
    # ==========================================
    
    p1 = Text("• O Limite da Autopoiese", font_size=28, color=RED_C, weight=BOLD)
    p1_sub = Text(
        "A autopoiese original é binária.\n"
        "Ou se está vivo, ou morto.\n"
        "Se o agente só reage quando falha,\n"
        "ele morre antes de aprender.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )
    
    p2 = Text("• A Solução: Adaptatividade", font_size=28, color=GREEN_C, weight=BOLD)
    p2_sub = Text(
        "A capacidade de monitorizar o\n"
        "próprio estado, sentir tendências\n"
        "de degradação e corrigir a rota\n"
        "ANTES da fronteira letal.", 
        font_size=22, color=LIGHT_GREY, line_spacing=1
    )

    bullets = VGroup(p1, p1_sub, p2, p2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    bullets.to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)

    self.play(FadeIn(p1, shift=LEFT*0.2), FadeIn(p1_sub, shift=LEFT*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: O Diagrama de Viabilidade à Direita
    # ==========================================
    
    # Criando as 3 zonas: Ideal (Verde), Precária (Amarela), Letal (Vermelha)
    zone_height = 4.0
    zone_ideal = Rectangle(width=1.5, height=zone_height, color=GREEN, fill_color=GREEN, fill_opacity=0.2, stroke_width=0)
    zone_prec = Rectangle(width=2.5, height=zone_height, color=YELLOW, fill_color=YELLOW, fill_opacity=0.2, stroke_width=0)
    zone_letal = Rectangle(width=1.5, height=zone_height, color=RED, fill_color=RED, fill_opacity=0.3, stroke_width=0)
    
    # Posicionando as zonas lado a lado
    zones = VGroup(zone_ideal, zone_prec, zone_letal).arrange(RIGHT, buff=0).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.2)
    
    # NOVO: O título "Restrições de Viabilidade" acima do diagrama
    viability_lbl = Text("Restrições de Viabilidade", font_size=20, color=WHITE, weight=BOLD).next_to(zones, UP, buff=0.4)
    
    lbl_ideal = Text("Zona\nIdeal", font_size=18, color=GREEN_C).move_to(zone_ideal).shift(UP * 1.5)
    lbl_prec = Text("Zona\nPrecária", font_size=18, color=YELLOW_D).move_to(zone_prec).shift(UP * 1.5)
    lbl_letal = Text("Zona\nLetal", font_size=18, color=RED_C).move_to(zone_letal).shift(UP * 1.5)
    
    # Linha fronteiriça da morte (Agora exatamente na borda entre a Precária e a Letal)
    death_line = DashedLine(start=zone_prec.get_corner(UR), end=zone_prec.get_corner(DR), color=RED)
    
    self.play(
        FadeIn(zones),
        FadeIn(viability_lbl), # Exibe o novo título
        FadeIn(lbl_ideal), FadeIn(lbl_prec), FadeIn(lbl_letal),
        Create(death_line)
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: Cenário 1 - Autopoiese sem Adaptatividade
    # ==========================================
    
    # Criamos uma versão simples da célula para representar o agente
    cell_membrane = Circle(radius=0.4, color=WHITE, stroke_width=3)
    cell_core = Dot(color=WHITE)
    agent = VGroup(cell_membrane, cell_core).move_to(zone_ideal.get_center() + DOWN * 0.5)

    self.play(FadeIn(agent, scale=0.5))
    
    # O agente move-se às cegas em direção ao perigo
    path_blind = Line(agent.get_right(), zone_letal.get_center() + DOWN * 0.5, color=GRAY)
    
    self.play(MoveAlongPath(agent, path_blind), run_time=2, rate_func=linear)
    
    # Ao cruzar a linha, morre
    cross_mark = Cross(agent, stroke_color=RED, stroke_width=6, scale_factor=0.8)
    
    lbl_death = Text("Morte\n(Tarde Demais)", font_size=18, color=RED, line_spacing=1).next_to(agent, DOWN)
    
    self.play(
        agent.animate.set_color(DARK_GRAY),
        Create(cross_mark),
        Write(lbl_death)
    )
    self.next_slide()

    # ==========================================
    # PARTE 4: Cenário 2 - Com Adaptatividade
    # ==========================================
    
    self.play(
        FadeOut(agent), FadeOut(cross_mark), FadeOut(lbl_death),
        FadeIn(p2, shift=LEFT*0.2), FadeIn(p2_sub, shift=LEFT*0.2)
    )

    # Renasce a célula na Zona Ideal, agora colorida
    agent_adapt = VGroup(
        Circle(radius=0.4, color=YELLOW, fill_color=GREEN_E, fill_opacity=0.5, stroke_width=3),
        Dot(color=BLUE)
    ).move_to(zone_ideal.get_center() + DOWN * 0.5)

    self.play(FadeIn(agent_adapt, scale=0.5))
    self.next_slide()

    # Move-se para a zona precária
    self.play(
        agent_adapt.animate.move_to(zone_prec.get_center() + DOWN * 0.5),
        run_time=1.5
    )
    
    # ADAPTATIVIDADE EM AÇÃO: O agente "sente" que a trajetória é perigosa
    alert_lbl = Text("! Tendência\nde Queda !", font_size=16, color=ORANGE, weight=BOLD, line_spacing=1).next_to(agent_adapt, UP)
    
    self.play(
        agent_adapt[0].animate.set_color(ORANGE).set_fill(ORANGE, opacity=0.5),
        Flash(agent_adapt, color=ORANGE),
        Write(alert_lbl)
    )
    self.next_slide()

    # O Agente CORRIGE a rota antes de morrer
    curved_path = CurvedArrow(
        start_point=agent_adapt.get_left(), 
        end_point=zone_ideal.get_right() + LEFT*0.3 + DOWN*0.5, 
        angle=TAU/4, 
        color=GREEN_C
    )
    
    correction_lbl = Text("Regulação", font_size=16, color=GREEN_C).next_to(curved_path, UP, buff=0.1)

    self.play(
        Create(curved_path),
        FadeIn(correction_lbl)
    )
    
    self.play(
        agent_adapt.animate.move_to(zone_ideal.get_center() + DOWN * 0.5),
        FadeOut(alert_lbl),
        run_time=1.5
    )
    
    # Volta a ficar saudável (verde/amarelo) e pulsa de vida
    self.play(
        agent_adapt[0].animate.set_color(YELLOW).set_fill(GREEN_E, opacity=0.5),
        Wiggle(agent_adapt, scale_value=1.1)
    )
    self.next_slide()

    # Transição final limpa
    self.play(*[FadeOut(mob) for mob in self.mobjects])
