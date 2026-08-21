from manim import *

def observacao(self):
    # --- Título ---
    title = Text("Observação", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: Carregando as Imagens Reais
    # ==========================================
    
    # Carregando e ajustando as alturas
    # O img_small tem uma altura menor para representar visualmente a perda de resolução
    img_rgb = ImageMobject("assets/rgb.png").scale_to_fit_height(2.5)
    img_gray = ImageMobject("assets/gray.png").scale_to_fit_height(2.5)
    img_small = ImageMobject("assets/smaller.png").scale_to_fit_height(1.8)

    # Posicionando as imagens em linha
    images = Group(img_rgb, img_gray, img_small).arrange(RIGHT, buff=1.5).shift(DOWN * 0.2)

    # Textos descritivos
    lbl_rgb = Text("RGB 320x240", font_size=20).next_to(img_rgb, UP, buff=0.4)
    lbl_gray = Text("Escala de Cinza", font_size=20).next_to(img_gray, UP, buff=0.4)
    lbl_small = Text("84x84 + Normalização [0, 1]", font_size=20).next_to(img_small, UP, buff=0.4)
    
    # Garantindo que o texto do smaller fique na mesma linha que os outros
    lbl_small.match_y(lbl_gray) 

    # Setas conectando as etapas
    arrow1 = Arrow(img_rgb.get_right(), img_gray.get_left(), buff=0.2, color=YELLOW)
    arrow2 = Arrow(img_gray.get_right(), img_small.get_left(), buff=0.2, color=YELLOW)

    # ==========================================
    # PARTE 2: Animações do Pipeline
    # ==========================================

    # Etapa 1: Mostra RGB original
    self.play(
        FadeIn(img_rgb, shift=UP*0.2), 
        Write(lbl_rgb)
    )
    self.next_slide()

    # Etapa 2: Mostra conversão para Cinza
    self.play(GrowArrow(arrow1))
    self.play(
        FadeIn(img_gray, shift=RIGHT * 0.5), 
        Write(lbl_gray),
        run_time=1.0
    )
    self.next_slide()

    # Etapa 3: Mostra redimensionamento (84x84)
    self.play(GrowArrow(arrow2))
    self.play(
        FadeIn(img_small, shift=RIGHT * 0.5), 
        Write(lbl_small),
        run_time=1.0
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: Conclusão
    # ==========================================
    
    # Texto explicativo final
    desc_text = Text(
        "Reduzir a complexidade visual mantendo a estrutura espacial", 
        font_size=24, 
        color=BLUE_C
    ).next_to(images, DOWN, buff=1.0)

    self.play(Write(desc_text))
    self.next_slide()

    # Transição final (Limpa a tela para o próximo slide)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
