from manim import *

def problema_recompensa(self):
    # --- Título ---
    title = Text("O Problema da Recompensa", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: A Imagem do Fantoche
    # ==========================================
    
    # Imagem atualizada e com tamanho levemente reduzido para dar espaço ao texto
    img = ImageMobject("assets/fantoche.png")
    img.height = 4.5
    img.to_edge(LEFT, buff=0.5).shift(DOWN * 0.2)

    self.play(FadeIn(img))
    self.next_slide()

    # ==========================================
    # PARTE 2: Textos baseados na Dissertação
    # ==========================================
    
    # Título da seção de texto
    analogy_title = Text("A Analogia do Fantoche", font_size=32, color=YELLOW, weight=BOLD)
    
    # Textos com quebras de linha mais curtas e fonte menor para não sair da tela
    b1 = Text(
        "• A solução esperada é embutida direto\n  no sinal pelo próprio projetista.", 
        font_size=20, 
        line_spacing=1
    )
    b2 = Text(
        "• O agente apenas repete uma rotina\n  para maximizar o número cegamente.", 
        font_size=20, 
        line_spacing=1
    )
    b3 = Text(
        "• Isso força uma solução fixa e impede a\n  descoberta de estratégias mais flexíveis.", 
        font_size=20, 
        line_spacing=1
    )
    b4 = Text(
        "• Comportamento monótono: Agir sempre\n  da mesma maneira, independente da situação,\n  não reflete inteligência.", 
        font_size=20, 
        color=RED_C, 
        line_spacing=1
    )

    # Agrupando e alinhando os textos à direita da imagem
    bullets = VGroup(analogy_title, b1, b2, b3, b4).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    bullets.next_to(img, RIGHT, buff=0.6)

    # ==========================================
    # PARTE 3: Animações dos Tópicos
    # ==========================================

    self.play(Write(analogy_title))
    self.next_slide()
    
    self.play(FadeIn(b1, shift=LEFT * 0.2))
    self.next_slide()
    
    self.play(FadeIn(b2, shift=LEFT * 0.2))
    self.next_slide()
    
    self.play(FadeIn(b3, shift=LEFT * 0.2))
    self.next_slide()

    self.play(FadeIn(b4, shift=LEFT * 0.2))
    self.next_slide()

    # Transição final (Limpa a tela para o próximo slide)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
