from manim import *

def intrinsic_reward(self):
    self.remove(*self.mobjects)
    
    # Lista de imagens na ordem exata solicitada
    slides_data = [
        {"path": "assets/intrinsic_reward_100.png", "title": "Recompensa e Precariedade (Força 100)"},
        {"path": "assets/intrinsic_reward_200.png", "title": "Recompensa e Precariedade (Força 200)"},
        {"path": "assets/intrinsic_reward_0.png", "title": "Recompensa e Precariedade (Força 0 - Normal)"},
        {"path": "assets/full_episode_spikes.png", "title": "Dinâmica de Recompensa no Fim do Treinamento"}
    ]

    for data in slides_data:
        # Título descritivo no topo
        title = Text(data["title"], font_size=32, weight=BOLD).to_edge(UP, buff=0.2)
        self.play(Write(title))

        # Carrega a imagem
        img = ImageMobject(data["path"])
        
        # 1. Força a largura a caber na tela com uma margem de segurança
        # (A tela do Manim tem ~14.22 de largura, 13 garante que não toca nas bordas)
        img.scale_to_fit_width(13.0)
        
        # 2. Se a imagem for mais "quadrada" e a altura ultrapassar o limite,
        # reduzimos pela altura para não sobrepor o título
        if img.height > 6.0:
            img.scale_to_fit_height(6.0)
            
        # Posiciona abaixo do título
        img.next_to(title, DOWN, buff=0.2)

        self.play(FadeIn(img, shift=UP * 0.2))
        
        # Pausa para o slide do manim-slides
        self.next_slide()

        # Limpa os elementos para o próximo loop
        self.play(FadeOut(title), FadeOut(img), run_time=0.4)
