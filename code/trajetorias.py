from manim import *

def trajetorias(self):
    # Limpa a tela para começar a apresentação dos gráficos
    self.remove(*self.mobjects)
    
    # Lista de imagens na ordem lógica (50, 100, 200, 0)
    slides_data = [
        {"path": "assets/trajectory_grid_50.png", "title": "Trajetórias no Espaço (Força 50)"},
        {"path": "assets/trajectory_grid_100.png", "title": "Trajetórias no Espaço (Força 100)"},
        {"path": "assets/trajectory_grid_200.png", "title": "Trajetórias no Espaço (Força 200)"},
        {"path": "assets/trajectory_grid_0.png", "title": "Trajetórias no Espaço (Força 0)"}
    ]

    for data in slides_data:
        # Título descritivo no topo
        title = Text(data["title"], font_size=32, weight=BOLD).to_edge(UP, buff=0.2)
        self.play(Write(title))

        # Carrega a imagem das trajetórias
        img = ImageMobject(data["path"])
        
        # 1. Como estas imagens são verticais, forçamos a altura máxima segura primeiro
        img.scale_to_fit_height(6.5)
        
        # 2. Margem de segurança de largura (caso o ecrã seja muito estreito)
        if img.width > 13.0:
            img.scale_to_fit_width(13.0)
            
        # Posiciona a imagem logo abaixo do título
        img.next_to(title, DOWN, buff=0.2)

        # Anima a entrada da grelha de trajetórias
        self.play(FadeIn(img, shift=UP * 0.2))
        
        # Pausa para o slide do manim-slides (permite mostrar as diferenças entre os agentes)
        self.next_slide()

        # Limpa os elementos para carregar a próxima imagem
        self.play(FadeOut(title), FadeOut(img), run_time=0.4)
