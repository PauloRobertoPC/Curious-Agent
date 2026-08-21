from manim import *

def motivacao_recompensa(self):
    # ==========================================
    # PARTE 1: Título e Relembrando a Pergunta
    # ==========================================
    title = Text("A Origem do Sinal", font_size=40, weight=BOLD).to_edge(UP, buff=0.2)
    author_ref = Text("(RYAN; DECI, 2000)", font_size=20, color=GRAY).next_to(title, DOWN, buff=0.1)
    
    self.play(Write(title), FadeIn(author_ref))
    
    q2_num = Text("2.", font_size=30, color=YELLOW, weight=BOLD)
    q2_text = Text(
        "Como transformar a luta contra a precariedade\nnum sinal de recompensa?", 
        font_size=22, line_spacing=1
    )
    question = VGroup(q2_num, q2_text).arrange(RIGHT, aligned_edge=UP, buff=0.2)
    question.next_to(author_ref, DOWN, buff=0.3)
    
    self.play(FadeIn(question))
    self.next_slide()
    
    # Minimiza a pergunta e alinha à esquerda
    self.play(
        question.animate.scale(0.7).to_edge(LEFT, buff=0.5).shift(UP * 2.0)
    )

    # ==========================================
    # FUNÇÃO AUXILIAR: Caixas Dinâmicas e Detalhadas
    # ==========================================
    def create_card(title_str, def_str, ex_str, logic_str, color):
        lbl_title = Text(title_str, font_size=26, color=color, weight=BOLD)
        
        # Definição formal
        lbl_def = Text(def_str, font_size=16, slant=ITALIC, line_spacing=1)
        
        # Exemplos
        lbl_ex = Text(ex_str, font_size=16, line_spacing=1)
        
        # A Lógica ("Faço X porque...")
        lbl_logic = Text(logic_str, font_size=18, color=color, weight=BOLD)
        
        # Agrupa os textos alinhados à esquerda
        content = VGroup(lbl_title, lbl_def, lbl_ex, lbl_logic).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        
        # Cria a caixa dinamicamente
        box = SurroundingRectangle(content, color=color, corner_radius=0.2, buff=0.4)
        box.set_fill(color, opacity=0.1)
        
        return VGroup(box, content)

    # ==========================================
    # PARTE 2: Extrínseca vs Intrínseca
    # ==========================================
    
    group_ext = create_card(
        "Motivação Extrínseca",
        "Realização de uma atividade como meio\npara atingir um objetivo externo.",
        "Ex: Estudar para tirar uma boa nota;\nTrabalhar para receber salário.",
        "“Eu faço X porque X me traz\n uma recompensa externa.”",
        RED_C
    )
    
    group_int = create_card(
        "Motivação Intrínseca",
        "Realização de uma atividade por\nseu valor inerente.",
        "Ex: Explorar uma floresta por curiosidade;\nJogar um jogo pelo desafio.",
        "“Eu faço X porque o próprio X\n é recompensador para mim.”",
        GREEN_C
    )
    
    # Organizando lado a lado e centralizando na tela
    cards = VGroup(group_ext, group_int).arrange(RIGHT, buff=0.5).shift(DOWN * 0.3)
    
    self.play(FadeIn(group_ext, shift=UP*0.2))
    self.next_slide() 
    
    self.play(FadeIn(group_int, shift=UP*0.2))
    self.next_slide() 

    # ==========================================
    # PARTE 3: A Rejeição da Extrínseca
    # ==========================================
    
    cross = Cross(group_ext, stroke_color=RED, stroke_width=8)
    self.play(Create(cross))
    self.next_slide()

    # ==========================================
    # PARTE 4: Centralizando a Intrínseca e a Nova Pergunta
    # ==========================================
    
    self.play(
        FadeOut(group_ext), FadeOut(cross), FadeOut(question),
        group_int.animate.move_to(UP * 0.4)
    )
    
    # A Grande Pergunta Transicional
    new_question = Text(
        "Como transformar Motivação Intrínseca\nem Recompensa?", 
        font_size=32, color=YELLOW, weight=BOLD, line_spacing=1
    ).next_to(group_int, DOWN, buff=0.6)
    
    self.play(Write(new_question))
    self.next_slide()

    # ==========================================
    # PARTE 5: A Resposta (Curiosidade - Pathak)
    # ==========================================
    
    # Limpamos o cartão e o título antigo, subimos a pergunta para o topo
    self.play(
        FadeOut(title), FadeOut(author_ref), FadeOut(group_int),
        new_question.animate.to_edge(UP, buff=0.5)
    )

    # Título da Curiosidade
    curiosity_title = Text("A Resposta: Curiosidade Artificial", font_size=36, color=BLUE_C, weight=BOLD)
    curiosity_author = Text("(PATHAK et al., 2017)", font_size=20, color=GRAY)
    curiosity_header = VGroup(curiosity_title, curiosity_author).arrange(DOWN, buff=0.1).next_to(new_question, DOWN, buff=0.8)

    self.play(FadeIn(curiosity_header, shift=UP*0.2))

    # Definições baseadas na dissertação
    c1 = Text("• Tendência em buscar experiências que revelem limitações\n  no conhecimento atual do agente.", font_size=24, line_spacing=1)
    c2 = Text("• Estados difíceis de prever tornam-se mais atrativos.", font_size=24)
    c3 = Text("• Recompensa Intrínseca = Erro de Predição", font_size=32, color=GREEN_C, weight=BOLD)

    curiosity_bullets = VGroup(c1, c2, c3).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(curiosity_header, DOWN, buff=0.8)

    self.play(FadeIn(c1, shift=LEFT*0.2))
    self.next_slide()

    self.play(FadeIn(c2, shift=LEFT*0.2))
    self.next_slide()

    # Destaca a formulação final (O gancho pro RND)
    self.play(FadeIn(c3, shift=UP*0.2), Flash(c3, color=GREEN_C, line_length=0.4))
    self.next_slide()

    # Limpa a tela para o próximo slide
    self.play(*[FadeOut(mob) for mob in self.mobjects])
