from manim import *

def rnd(self):
    # ==========================================
    # PARTE 1: Título e as Duas Redes Neurais
    # ==========================================
    title = Text("Random Network Distillation (RND)", font_size=40, weight=BOLD).to_edge(UP, buff=0.3)
    self.play(Write(title))

    # Função auxiliar para desenhar uma rede neural simples
    def create_network(layers, node_color, edge_color):
        net = VGroup()
        for n_nodes in layers:
            layer = VGroup(*[Circle(radius=0.15, color=node_color, fill_color=node_color, fill_opacity=0.2) for _ in range(n_nodes)])
            layer.arrange(DOWN, buff=0.2)
            net.add(layer)
        net.arrange(RIGHT, buff=0.8)
        
        edges = VGroup()
        for i in range(len(layers)-1):
            for n1 in net[i]:
                for n2 in net[i+1]:
                    edges.add(Line(n1.get_right(), n2.get_left(), stroke_width=1.5, color=edge_color, stroke_opacity=0.5))
        return VGroup(edges, net)

    # Criando a Rede Alvo (Fixa)
    net_alvo = create_network([3, 4, 2], BLUE, LIGHT_GREY)
    lbl_alvo = Text("Rede Alvo\n(Pesos Fixos / Congelados)", font_size=20, color=BLUE_C).next_to(net_alvo, UP, buff=0.2)
    group_alvo = VGroup(lbl_alvo, net_alvo)

    # Criando a Rede Preditora (Treinável)
    net_pred = create_network([3, 4, 2], GREEN, LIGHT_GREY)
    lbl_pred = Text("Rede Preditora\n(Treinável / Aprende)", font_size=20, color=GREEN_C).next_to(net_pred, DOWN, buff=0.2)
    group_pred = VGroup(net_pred, lbl_pred)

    # Organizando as redes na tela (Movido para BAIXO para não sobrepor o título)
    networks = VGroup(group_alvo, group_pred).arrange(DOWN, buff=0.8).shift(LEFT * 2.5 + DOWN * 0.2)
    
    self.play(FadeIn(group_alvo, shift=DOWN*0.2))
    self.play(FadeIn(group_pred, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 2: Abstração Matemática f(x) e f'(x)
    # ==========================================
    
    # Caixas das funções matemáticas
    box_alvo = RoundedRectangle(width=3, height=1.5, color=BLUE, fill_color=BLUE, fill_opacity=0.1).move_to(group_alvo)
    math_alvo = MathTex(r"f(x)", font_size=48, color=BLUE_C).move_to(box_alvo)
    
    box_pred = RoundedRectangle(width=3, height=1.5, color=GREEN, fill_color=GREEN, fill_opacity=0.1).move_to(group_pred)
    math_pred = MathTex(r"\hat{f}(x)", font_size=48, color=GREEN_C).move_to(box_pred)

    # Transforma as redes complexas nas funções matemáticas simplificadas
    self.play(
        Transform(group_alvo, VGroup(box_alvo, math_alvo)),
        Transform(group_pred, VGroup(box_pred, math_pred))
    )
    
    lbl_f = Text("Função Alvo", font_size=20, color=BLUE_C).next_to(box_alvo, UP)
    lbl_f_hat = Text("Função Preditora", font_size=20, color=GREEN_C).next_to(box_pred, DOWN)
    
    self.play(FadeIn(lbl_f), FadeIn(lbl_f_hat))
    self.next_slide()

    # ==========================================
    # PARTE 3: Primeira Iteração - O Estado Frequente (x)
    # ==========================================
    
    # Input x (Agrupado com o texto ANTES de mover para a borda esquerda para não vazar)
    input_x = Square(side_length=0.6, color=YELLOW, fill_color=YELLOW, fill_opacity=0.5)
    lbl_x = Text("Estado Frequente (x)", font_size=20, color=YELLOW)
    
    group_input_x = VGroup(lbl_x, input_x).arrange(DOWN, buff=0.2)
    group_input_x.to_edge(LEFT, buff=0.5).shift(UP * 0.5)
    
    self.play(FadeIn(group_input_x))
    
    # x entra nas duas redes
    x_copy1 = input_x.copy()
    x_copy2 = input_x.copy()
    
    self.play(
        x_copy1.animate.move_to(box_alvo.get_left()),
        x_copy2.animate.move_to(box_pred.get_left()),
        run_time=1
    )
    self.play(FadeOut(x_copy1), FadeOut(x_copy2))

    # Saídas contidas para não vazar a tela
    out_alvo = Dot(color=BLUE, radius=0.15).move_to(RIGHT * 1.0 + UP * 1.0)
    out_pred = Dot(color=GREEN, radius=0.15).move_to(RIGHT * 2.5 + DOWN * 0.5) 
    
    arrow_out_alvo = Arrow(box_alvo.get_right(), out_alvo.get_left(), color=BLUE, buff=0.1)
    arrow_out_pred = Arrow(box_pred.get_right(), out_pred.get_left(), color=GREEN, buff=0.1)

    self.play(GrowArrow(arrow_out_alvo), FadeIn(out_alvo))
    self.play(GrowArrow(arrow_out_pred), FadeIn(out_pred))

    # Calcula o Erro ancorado na margem direita
    error_line = DashedLine(out_alvo.get_center(), out_pred.get_center(), color=RED)
    
    error_math = MathTex(r"Erro = (f(x) - \hat{f}(x))^2", font_size=28, color=RED)
    error_lbl = Text("ERRO ALTO = Alta Recompensa", font_size=20, color=RED)
    
    error_group = VGroup(error_math, error_lbl).arrange(DOWN, buff=0.2)
    error_group.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)
    
    self.play(Create(error_line))
    self.play(FadeIn(error_group, shift=LEFT * 0.2))
    self.next_slide()

    # ==========================================
    # PARTE 4: O Aprendizado (Treinamento)
    # ==========================================
    
    learning_lbl = Text("Aprendendo...", font_size=24, color=GREEN).next_to(box_pred, DOWN, buff=0.2)
    
    # A preditora pisca para indicar treinamento e o ponto verde se aproxima do azul
    self.play(
        FadeOut(lbl_f_hat),
        FadeIn(learning_lbl),
        Flash(box_pred, color=GREEN, line_length=0.3, num_lines=12),
        out_pred.animate.move_to(RIGHT * 1.2 + UP * 0.5), # Aproxima de f(x)
        UpdateFromFunc(error_line, lambda l: l.put_start_and_end_on(out_alvo.get_center(), out_pred.get_center())),
        run_time=2
    )
    
    # Atualiza o texto do erro
    new_error_lbl = Text("ERRO BAIXO = Baixa Recompensa", font_size=20, color=GRAY).move_to(error_lbl)
    
    self.play(
        Transform(error_lbl, new_error_lbl),
        error_line.animate.set_color(GRAY),
        error_math.animate.set_color(GRAY),
        FadeOut(learning_lbl),
        FadeIn(lbl_f_hat)
    )
    self.next_slide()

    # ==========================================
    # PARTE 5: A Chegada do Estado Raro (y)
    # ==========================================
    
    # Limpa as saídas do x
    self.play(
        FadeOut(group_input_x),
        FadeOut(out_alvo), FadeOut(out_pred),
        FadeOut(arrow_out_alvo), FadeOut(arrow_out_pred),
        FadeOut(error_line), FadeOut(error_group)
    )

    # Input y (Agrupado com o texto ANTES de mover para a borda)
    input_y = Triangle(color=PURPLE, fill_color=PURPLE, fill_opacity=0.5)
    lbl_y = Text("Estado Raro (y)", font_size=20, color=PURPLE)
    
    group_input_y = VGroup(lbl_y, input_y).arrange(DOWN, buff=0.2)
    group_input_y.to_edge(LEFT, buff=0.5).shift(UP * 0.5)
    
    self.play(FadeIn(group_input_y))
    
    # y entra nas duas redes
    y_copy1 = input_y.copy()
    y_copy2 = input_y.copy()
    
    self.play(
        y_copy1.animate.move_to(box_alvo.get_left()),
        y_copy2.animate.move_to(box_pred.get_left()),
        run_time=1
    )
    self.play(FadeOut(y_copy1), FadeOut(y_copy2))

    # Novas Saídas para y (A rede preditora não conhece y, então erra feio)
    out_alvo_y = Dot(color=BLUE, radius=0.15).move_to(RIGHT * 1.0 + UP * 1.0)
    out_pred_y = Dot(color=GREEN, radius=0.15).move_to(RIGHT * 3.0 + DOWN * 1.0)
    
    arrow_y_alvo = Arrow(box_alvo.get_right(), out_alvo_y.get_left(), color=BLUE, buff=0.1)
    arrow_y_pred = Arrow(box_pred.get_right(), out_pred_y.get_left(), color=GREEN, buff=0.1)

    self.play(GrowArrow(arrow_y_alvo), FadeIn(out_alvo_y))
    self.play(GrowArrow(arrow_y_pred), FadeIn(out_pred_y))

    # Erro gigantesco gera CURIOSIDADE ancorado à direita
    error_line_y = DashedLine(out_alvo_y.get_center(), out_pred_y.get_center(), color=RED)
    
    surprise_text = Text("CURIOSIDADE!", font_size=32, color=YELLOW, weight=BOLD)
    error_math_y = MathTex(r"Erro = (f(y) - \hat{f}(y))^2", font_size=28, color=RED)
    error_lbl_y = Text("ERRO ALTO = Alta Recompensa", font_size=20, color=RED)
    
    error_group_y = VGroup(surprise_text, error_math_y, error_lbl_y).arrange(DOWN, buff=0.2)
    error_group_y.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

    self.play(Create(error_line_y))
    self.play(FadeIn(error_group_y, shift=LEFT * 0.2))
    self.play(Flash(surprise_text, color=YELLOW, line_length=0.5))
    self.next_slide()

    # ==========================================
    # PARTE 6: Conclusão RND
    # ==========================================
    
    # Caixa de resumo dinâmica (SurroundingRectangle) ancorada na borda inferior
    summary_txt = Text(
        "Frequência melhora a predição = Recompensa Cai.\n"
        "Estados Raros mantêm o Erro Alto = O Agente é recompensado por explorar!",
        font_size=20, line_spacing=1
    )
    summary_box = SurroundingRectangle(summary_txt, color=WHITE, corner_radius=0.2, buff=0.3)
    summary_box.set_fill(BLACK, opacity=0.9) 
    
    summary_group = VGroup(summary_box, summary_txt).to_edge(DOWN, buff=0.0)
    
    self.play(FadeIn(summary_group, shift=UP*0.2))
    self.next_slide()

    # Limpa a tela para a próxima cena
    self.play(*[FadeOut(mob) for mob in self.mobjects])
