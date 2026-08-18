from manim import *

def ponte(self):
    # --- Título ---
    title = Text("O Limite da IA Tradicional", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: Resumo do Beco Sem Saída
    # ==========================================
    
    # Ponto 1: O Fantoche (Falta de Autonomia)
    p1_title = Text("1. Falta de Autonomia", font_size=28, color=RED_C, weight=BOLD)
    p1_desc = Text(
        "O agente é monótono: apenas executa o que o\nprojetista mandou através da função de recompensa.", 
        font_size=24,
        line_spacing=1
    )
    group1 = VGroup(p1_title, p1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

    # Ponto 2: O Quarto Chinês (Falta de Compreensão)
    p2_title = Text("2. Falta de Compreensão", font_size=28, color=RED_C, weight=BOLD)
    p2_desc = Text(
        "O agente manipula números mecanicamente,\nmas é incapaz de compreender o que está fazendo.", 
        font_size=24,
        line_spacing=1
    )
    group2 = VGroup(p2_title, p2_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

    # Agrupando as duas falhas
    flaws = VGroup(group1, group2).arrange(DOWN, aligned_edge=LEFT, buff=0.8).next_to(title, DOWN, buff=1.0)

    self.play(FadeIn(group1, shift=UP * 0.2))
    self.next_slide()

    self.play(FadeIn(group2, shift=UP * 0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: A Pergunta de Transição
    # ==========================================
    
    # Limpamos a tela para dar um respiro e focar na pergunta
    self.play(FadeOut(flaws))

    question1 = Text("Se olharmos para o mundo...", font_size=32)
    question2 = Text("Onde existe inteligência real?", font_size=36, color=BLUE_C, weight=BOLD)
    
    q_group = VGroup(question1, question2).arrange(DOWN, buff=0.5)

    self.play(Write(question1))
    self.play(FadeIn(question2, shift=UP * 0.2))
    self.next_slide()

    # ==========================================
    # PARTE 3: A Resposta / Hipótese Central
    # ==========================================
    
    # A resposta para a pergunta
    answer = Text("Nós a vemos na VIDA.", font_size=40, color=GREEN_C, weight=BOLD)
    
    self.play(
        FadeOut(question1),
        Transform(question2, answer) # A pergunta se transforma na resposta
    )
    self.next_slide()

    # O Clímax: A Hipótese da Dissertação
    hipotese_lbl = Text("HIPÓTESE CENTRAL", font_size=24, color=YELLOW, weight=BOLD).shift(UP * 1)
    
    # Usando t2c (text to color) para destacar as palavras-chave na mesma frase
    hipotese_text = Text(
        "A Inteligência é um fenômeno\nemergente da vida.", 
        font_size=44, 
        weight=BOLD,
        t2c={"Inteligência": BLUE_C, "emergente da vida": GREEN_C},
        line_spacing=1
    ).next_to(hipotese_lbl, DOWN, buff=0.5)

    self.play(
        FadeOut(question2), # (que agora é o 'answer')
        FadeIn(hipotese_lbl, shift=UP * 0.2),
        Write(hipotese_text)
    )
    self.next_slide()

    # Transição final (Limpa a tela para o próximo passo)
    self.play(*[FadeOut(mob) for mob in self.mobjects])
