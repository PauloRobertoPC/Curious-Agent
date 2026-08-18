from manim import *

def chines(self):
    # --- Título ---
    title = Text("O Quarto Chinês (John Searle, 1980)", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # Subtítulo (A refutação)
    subtitle = Text("Uma refutação direta ao Teste de Turing", font_size=28, color=RED).next_to(title, DOWN, buff=0.2)
    self.play(Write(subtitle))
    self.next_slide()

    # ==========================================
    # PARTE 1: O Cenário do Quarto
    # ==========================================
    
    # O Quarto
    room = Rectangle(width=4.5, height=3.5, color=WHITE, stroke_width=4)
    room.shift(DOWN * 0.5)
    room_label = Text("Quarto Fechado", font_size=24, color=GRAY).next_to(room, UP, buff=0.2)

    # O Operador (Dentro do quarto)
    person = Dot(radius=0.3, color=BLUE).move_to(room.get_center() + LEFT * 0.8)
    person_lbl1 = Text("Operador", font_size=20, color=BLUE).next_to(person, DOWN, buff=0.1)
    person_lbl2 = Text("(Zero compreensão)", font_size=16, color=LIGHT_GREY).next_to(person_lbl1, DOWN, buff=0.1)
    person_group = VGroup(person, person_lbl1, person_lbl2)
    
    # O Manual de Regras
    book = Rectangle(width=1.0, height=1.4, color=YELLOW, fill_opacity=0.1).move_to(room.get_center() + RIGHT * 0.8)
    book_lines = VGroup(*[Line(LEFT*0.3, RIGHT*0.3, stroke_width=2) for _ in range(4)]).arrange(DOWN, buff=0.15).move_to(book)
    book_lbl = Text("Manual de\nRegras", font_size=16, color=YELLOW).next_to(book, DOWN, buff=0.2)
    book_group = VGroup(book, book_lines, book_lbl)

    self.play(Create(room), Write(room_label))
    self.play(FadeIn(person_group, shift=UP*0.2), FadeIn(book_group, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: O Fluxo de Informação
    # ==========================================
    
    # Elementos externos
    input_lbl = Text("Entrada\n(Símbolos)", font_size=20).next_to(room, LEFT, buff=1.0)
    output_lbl = Text("Saída\n(Respostas)", font_size=20).next_to(room, RIGHT, buff=1.0)

    self.play(FadeIn(input_lbl), FadeIn(output_lbl))
    
    # Animação da Entrada (Usando formas abstractas para evitar erros de fonte com caracteres reais)
    msg_in = Square(side_length=0.4, color=GREEN, fill_opacity=0.5).next_to(input_lbl, RIGHT)
    
    self.play(msg_in.animate.next_to(person, LEFT, buff=0.3))
    self.next_slide()

    # Consulta ao Manual
    rule_text = Text("Regra: Se [ ■ ] então responda [ ▲ ]", font_size=20, color=YELLOW).next_to(book, UP, buff=0.5)
    
    self.play(Indicate(book_group, color=YELLOW, scale_factor=1.1))
    self.play(Write(rule_text))
    self.next_slide()

    # Geração da Saída
    msg_out = Triangle(color=RED, fill_opacity=0.5).scale(0.3).next_to(person, RIGHT, buff=0.3)
    
    self.play(
        TransformFromCopy(msg_in, msg_out),
        FadeOut(msg_in)
    )
    self.play(msg_out.animate.next_to(output_lbl, LEFT))
    self.next_slide()

    # ==========================================
    # PARTE 3: A Conclusão Filosófica
    # ==========================================
    
    # Apaga o diagrama suavemente
    self.play(
        FadeOut(room), FadeOut(room_label), FadeOut(person_group), FadeOut(book_group),
        FadeOut(input_lbl), FadeOut(output_lbl), FadeOut(msg_out), FadeOut(rule_text),
        FadeOut(subtitle)
    )

    # Conclusões (Usando VGroup e aligned_edge=LEFT para evitar erros de alinhamento)
    c1 = Text("Sintaxe não é Semântica:", font_size=32, color=RED_C, weight=BOLD)
    c2 = Text("Manipular regras perfeitamente não significa compreender o significado.", font_size=24)

    c3 = Text("A Ilusão do Comportamento:", font_size=32, color=BLUE_C, weight=BOLD)
    c4 = Text("O operador passa no Teste de Turing, mas é apenas um fantoche mecânico...", font_size=24)
    c5 = Text("Incapaz de explicar o que lhe foi perguntado.", font_size=24, color=YELLOW)

    # Organizando os textos
    group1 = VGroup(c1, c2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    group2 = VGroup(c3, c4, c5).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

    conclusions = VGroup(group1, group2).arrange(DOWN, aligned_edge=LEFT, buff=0.8).shift(DOWN*0.3)

    self.play(FadeIn(group1, shift=UP*0.2))
    self.next_slide()
    
    self.play(FadeIn(group2, shift=UP*0.2))
    self.next_slide()

    # Limpa a tela para a Grande Ponte Filosófica
    self.play(*[FadeOut(mob) for mob in self.mobjects])
