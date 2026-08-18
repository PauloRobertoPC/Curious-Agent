from manim import *

def turing(self):
    # --- Título ---
    title = Text("O Teste de Turing (1950)", font_size=40, weight=BOLD).to_edge(UP)
    self.play(Write(title))
    self.next_slide()

    # ==========================================
    # PARTE 1: A Premissa de Turing
    # ==========================================
    
    quote1 = Text('"Pode uma máquina pensar?"', font_size=32, color=LIGHT_GREY)
    quote2 = Text('Substituído por: "Pode uma máquina imitar?"', font_size=32, color=YELLOW)
    
    quotes = VGroup(quote1, quote2).arrange(DOWN, buff=0.4).shift(UP * 1.5)

    self.play(FadeIn(quote1, shift=UP * 0.2))
    self.next_slide()
    
    self.play(FadeIn(quote2, shift=UP * 0.2))
    self.next_slide()

    # Movemos as citações para cima para dar espaço ao diagrama
    self.play(quotes.animate.scale(0.7).to_edge(UP, buff=1.2))

    # ==========================================
    # PARTE 2: O Diagrama do Teste
    # ==========================================
    
    # Avaliador (Juiz) à esquerda
    judge = Dot(radius=0.4, color=BLUE)
    judge_lbl = Text("Avaliador", font_size=24).next_to(judge, DOWN)
    judge_group = VGroup(judge, judge_lbl).to_edge(LEFT, buff=1.5).shift(DOWN * 1)

    # Parede (A "cegueira" do teste)
    wall = DashedLine(start=UP*1, end=DOWN*3, color=GRAY, stroke_width=4).shift(LEFT * 0.5)

    # Máquina e Humano à direita
    machine = Square(side_length=0.7, color=RED, fill_opacity=0.3)
    machine_lbl = Text("Máquina", font_size=24).next_to(machine, RIGHT, buff=0.3)
    machine_group = VGroup(machine, machine_lbl).to_edge(RIGHT, buff=2).shift(UP * 0.2)

    human = Dot(radius=0.4, color=GREEN)
    human_lbl = Text("Humano", font_size=24).next_to(human, RIGHT, buff=0.3)
    human_group = VGroup(human, human_lbl).to_edge(RIGHT, buff=2).shift(DOWN * 2.2)

    # Anima a entrada do cenário
    self.play(
        FadeIn(judge_group),
        Create(wall),
        FadeIn(machine_group),
        FadeIn(human_group)
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: O Jogo da Imitação (A Troca de Mensagens)
    # ==========================================
    
    # Mensagem do Juiz para a Máquina
    q_arrow1 = Arrow(judge.get_right(), machine.get_left(), color=WHITE, buff=0.2)
    a_arrow1 = Arrow(machine.get_left(), judge.get_right(), color=RED, buff=0.2).shift(DOWN*0.2)
    
    self.play(GrowArrow(q_arrow1))
    self.play(GrowArrow(a_arrow1))
    
    # Mensagem do Juiz para o Humano
    q_arrow2 = Arrow(judge.get_right(), human.get_left(), color=WHITE, buff=0.2)
    a_arrow2 = Arrow(human.get_left(), judge.get_right(), color=GREEN, buff=0.2).shift(DOWN*0.2)

    self.play(GrowArrow(q_arrow2))
    self.play(GrowArrow(a_arrow2))
    self.next_slide()

    # A Dúvida do Juiz
    question_mark = Text("?", font_size=60, color=YELLOW, weight=BOLD).next_to(judge, UP, buff=0.2)
    self.play(Write(question_mark), Flash(question_mark, color=YELLOW))
    self.next_slide()

    # ==========================================
    # PARTE 4: A Conclusão (O Conceito)
    # ==========================================
    
    # Oculta o diagrama suavemente para focar na conclusão
    self.play(
        FadeOut(judge_group), FadeOut(wall), FadeOut(machine_group), FadeOut(human_group),
        FadeOut(q_arrow1), FadeOut(a_arrow1), FadeOut(q_arrow2), FadeOut(a_arrow2), FadeOut(question_mark)
    )

    concept_title = Text("O Paradigma Tradicional da IA:", font_size=32, color=BLUE_C)
    
    # Correção: Removido o argumento 'alignment' que estava causando o TypeError
    concept_text = Text(
        "Se o comportamento externo (output) é\nindistinguível, o agente é considerado inteligente.", 
        font_size=28, 
        line_spacing=1
    )

    concept_group = VGroup(concept_title, concept_text).arrange(DOWN, buff=0.5).shift(DOWN * 0.5)

    self.play(FadeIn(concept_group, shift=UP*0.2))
    self.next_slide()

    # Limpa a tela para o Quarto Chinês
    self.play(*[FadeOut(mob) for mob in self.mobjects])
