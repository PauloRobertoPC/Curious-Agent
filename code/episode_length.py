from manim import *

def episode_length(self):
    # Limpa a tela para começar a apresentação dos gráficos
    self.remove(*self.mobjects)
    
    # Lista de imagens na ordem exata solicitada
    slides_data = [
        {"path": "assets/episode_length_50.png", "title": "Duração do Episódio (Força 50)"},
        {"path": "assets/episode_length_100.png", "title": "Duração do Episódio (Força 100)"},
        {"path": "assets/episode_length_200.png", "title": "Duração do Episódio (Força 200)"},
        {"path": "assets/episode_length_0.png", "title": "Duração do Episódio (Força 0)"}
    ]

    for data in slides_data:
        # Título descritivo no topo
        title = Text(data["title"], font_size=32, weight=BOLD).to_edge(UP, buff=0.2)
        self.play(Write(title))

        # Carrega a imagem do gráfico
        img = ImageMobject(data["path"])
        
        # 1. Força a largura a caber na tela com uma margem de segurança (13 unidades)
        img.scale_to_fit_width(13.0)
        
        # 2. Se a imagem ultrapassar a altura máxima após o ajuste de largura, limitamos pela altura
        if img.height > 6.0:
            img.scale_to_fit_height(6.0)
            
        # Posiciona a imagem logo abaixo do título
        img.next_to(title, DOWN, buff=0.2)

        # Anima a entrada do gráfico
        self.play(FadeIn(img, shift=UP * 0.2))
        
        # Pausa para o slide do manim-slides (permite a explicação antes de passar ao próximo)
        self.next_slide()

        # Limpa os elementos para carregar o gráfico seguinte
        self.play(FadeOut(title), FadeOut(img), run_time=0.4)
