from manim import *

def gradcam(self):
    # Limpa a tela para começar a apresentação dos mapas de calor
    self.remove(*self.mobjects)
    
    # Lista de imagens na ordem lógica (50, 100, 200, 0)
    slides_data = [
        {"path": "assets/gradcam_50.png", "title": "Atenção Visual - Grad-CAM (Força 50)"},
        {"path": "assets/gradcam_100.png", "title": "Atenção Visual - Grad-CAM (Força 100)"},
        {"path": "assets/gradcam_200.png", "title": "Atenção Visual - Grad-CAM (Força 200)"},
        {"path": "assets/gradcam_0.png", "title": "Atenção Visual - Grad-CAM (Força 0)"}
    ]

    for data in slides_data:
        # Título descritivo no topo
        title = Text(data["title"], font_size=32, weight=BOLD).to_edge(UP, buff=0.2)
        self.play(Write(title))

        # Carrega a imagem do Grad-CAM
        img = ImageMobject(data["path"])
        
        # 1. Ajusta para a altura máxima segura, deixando espaço para o título
        img.scale_to_fit_height(6.2)
        
        # 2. Margem de segurança de largura (caso a imagem seja mais larga que a proporção da tela)
        if img.width > 13.0:
            img.scale_to_fit_width(13.0)
            
        # Posiciona a imagem logo abaixo do título
        img.next_to(title, DOWN, buff=0.2)

        # Anima a entrada do mapa de atenção
        self.play(FadeIn(img, shift=UP * 0.2))
        
        # Pausa para o slide do manim-slides 
        # (Permite-lhe explicar à banca a diferença de foco entre as 3 colunas)
        self.next_slide()

        # Limpa os elementos para carregar a próxima imagem
        self.play(FadeOut(title), FadeOut(img), run_time=0.4)
