from manim import *

def recompensa_extrinsica(self):
    # --- Título ---
    title = Text("Recompensa Extrínseca", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    subtitle = Text("Sinal de recompensa fornecido pelo ambiente", font_size=28, color=GRAY).next_to(title, DOWN, buff=0.2)
    self.play(Write(subtitle))
    self.next_slide()

    # ==========================================
    # FUNÇÃO AUXILIAR: Criar os "Cartões" de Recompensa
    # ==========================================
    def create_reward_card(title_text, icon, value_text, value_color, box_color):
        # Caixa principal
        box = RoundedRectangle(corner_radius=0.2, height=3.5, width=3.5, color=box_color, fill_opacity=0.1)
        
        # Título superior
        lbl_title = Text(title_text, font_size=24).next_to(box.get_top(), DOWN, buff=0.3)
        
        # Ícone centralizado
        icon.move_to(box.get_center())
        
        # Valor de recompensa na parte inferior
        lbl_value = Text(value_text, font_size=48, color=value_color, weight=BOLD).next_to(box.get_bottom(), UP, buff=0.3)
        
        return VGroup(box, lbl_title, icon, lbl_value)

    # ==========================================
    # PARTE 1: Criando os Ícones
    # ==========================================
    
    # Ícone do Medikit (Caixa branca com cruz verde)
    medikit_bg = Square(side_length=0.8, fill_color=LIGHT_GRAY, fill_opacity=1, stroke_width=0)
    medikit_cross = Cross(stroke_color=GREEN, scale_factor=0.25, stroke_width=8).move_to(medikit_bg)
    icon_medikit = VGroup(medikit_bg, medikit_cross)

    # Ícone do Tempo/Tick (Relógio minimalista)
    clock_circle = Circle(radius=0.4, color=WHITE, stroke_width=4)
    clock_hand1 = Line(ORIGIN, UP*0.25, color=WHITE, stroke_width=3)
    clock_hand2 = Line(ORIGIN, RIGHT*0.15, color=WHITE, stroke_width=3)
    icon_clock = VGroup(clock_circle, clock_hand1, clock_hand2)

    # ==========================================
    # PARTE 2: Montando os Cartões
    # ==========================================
    
    card_medikit = create_reward_card(
        "Coletar Medikit", 
        icon_medikit, 
        "+ 1.0", 
        GREEN, 
        GREEN_C
    )
    
    card_tick = create_reward_card(
        "Sobreviver (Por Tick)", 
        icon_clock, 
        "+ 0.01", 
        YELLOW, 
        YELLOW_C
    )

    # Agrupando lado a lado
    cards = VGroup(card_medikit, card_tick).arrange(RIGHT, buff=1.0).shift(DOWN * 0.2)

    # ==========================================
    # PARTE 3: Animações
    # ==========================================

    # Apresenta a recompensa primária (Medikit)
    self.play(FadeIn(card_medikit, shift=UP * 0.5))
    self.next_slide()

    # Apresenta a recompensa de sobrevivência contínua (Tick)
    self.play(FadeIn(card_tick, shift=UP * 0.5))
    self.next_slide()

    # Equação final demonstrando a recompensa total no instante 't'
    eq_text = MathTex(
        r"R_t = R_{\text{medikit}} + R_{\text{tick}}", 
        font_size=36
    ).next_to(cards, DOWN, buff=0.6)
    
    self.play(Write(eq_text))
    self.next_slide()

    # Transição final (Limpa a tela)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
