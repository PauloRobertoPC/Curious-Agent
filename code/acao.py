from manim import *

def acao(self):
    # --- Título ---
    title = Text("Espaço de Ação", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    subtitle = Text("Espaço de Ação Discreto (4 ações possíveis)", font_size=28, color=YELLOW).next_to(title, DOWN, buff=0.3)
    self.play(Write(subtitle))
    self.next_slide()

    # ==========================================
    # FUNÇÃO AUXILIAR: Criar os "Cartões" de Ação
    # ==========================================
    def create_action_card(text, icon, color):
        # Cria uma caixa com bordas arredondadas (estilo botão)
        box = RoundedRectangle(corner_radius=0.2, height=2.5, width=2.2, color=color, fill_opacity=0.1)
        
        # O texto fica na parte inferior da caixa
        label = Text(text, font_size=24).next_to(box.get_bottom(), UP, buff=0.3)
        
        # O ícone fica centralizado, um pouco acima do texto
        icon.move_to(box.get_center()).shift(UP * 0.2)
        
        return VGroup(box, icon, label)

    # ==========================================
    # PARTE 1: Criando os Ícones
    # ==========================================
    
    # Setas (Frente, Esquerda, Direita)
    arrow_esq = Arrow(start=RIGHT*0.6, end=LEFT*0.6, buff=0, color=BLUE, stroke_width=8)
    arrow_frente = Arrow(start=DOWN*0.6, end=UP*0.6, buff=0, color=GREEN, stroke_width=8)
    arrow_dir = Arrow(start=LEFT*0.6, end=RIGHT*0.6, buff=0, color=BLUE, stroke_width=8)
    
    # Ícone "Parado" (Símbolo de Pause: duas barras verticais)
    bar1 = Rectangle(width=0.15, height=0.7, fill_color=GRAY, fill_opacity=1, stroke_width=0).shift(LEFT * 0.15)
    bar2 = Rectangle(width=0.15, height=0.7, fill_color=GRAY, fill_opacity=1, stroke_width=0).shift(RIGHT * 0.15)
    icon_parado = VGroup(bar1, bar2)

    # ==========================================
    # PARTE 2: Montando os Cartões
    # ==========================================
    card_esq = create_action_card("Esquerda", arrow_esq, BLUE)
    card_frente = create_action_card("Frente", arrow_frente, GREEN)
    card_dir = create_action_card("Direita", arrow_dir, BLUE)
    card_parado = create_action_card("Parado", icon_parado, GRAY)

    # Agrupando todos os cartões lado a lado
    cards = VGroup(card_esq, card_frente, card_dir, card_parado).arrange(RIGHT, buff=0.5).shift(DOWN * 0.2)

    # ==========================================
    # PARTE 3: Animações
    # ==========================================

    # A entrada dos cartões com um efeito cascata (LaggedStart) muito elegante
    self.play(
        LaggedStart(
            FadeIn(card_esq, shift=UP * 0.5),
            FadeIn(card_frente, shift=UP * 0.5),
            FadeIn(card_dir, shift=UP * 0.5),
            FadeIn(card_parado, shift=UP * 0.5),
            lag_ratio=0.25
        ),
        run_time=2.0
    )
    self.next_slide()

    # Mostrando a representação matemática do Espaço de Ação para RL
    math_text = MathTex(r"A_t \in \{0, 1, 2, 3\}", font_size=36).next_to(cards, DOWN, buff=0.8)
    math_desc = Text("Vetor enviado ao ambiente a cada passo", font_size=20, color=LIGHT_GREY).next_to(math_text, DOWN, buff=0.2)
    
    math_group = VGroup(math_text, math_desc)

    self.play(FadeIn(math_group, shift=UP * 0.2))
    self.next_slide()

    # Transição final (Limpa a tela para o próximo slide)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
