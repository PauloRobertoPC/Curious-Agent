from manim import *

def title(self):
    title = Text(
        "PRECARIEDADE É TUDO QUE VOCÊ PRECISA:",
        font_size=44,
        weight=BOLD,
    )

    subtitle1 = Text(
        "O CORPO COMO OBJETIVO DE APRENDIZADO ",
        font_size=32,
        weight=BOLD,
    )

    subtitle2 = Text(
        "POR REFORÇO EM AGENTES VIRTUAIS",
        font_size=32,
        weight=BOLD,
    )

    title_group = VGroup(
        title,
        subtitle1,
        subtitle2,
    ).arrange(
        DOWN,
        buff=0.25,
    )

    title_group.to_edge(UP, buff=1.5)

    self.play(
        Write(title),
        run_time=1.0,
    )

    self.play(
        Write(subtitle1),
        run_time=1.0,
    )

    self.play(
        Write(subtitle2),
        run_time=1.0,
    )

    author = Text(
        "Paulo Roberto Pinto Costa",
        font_size=26,
    )

    program = Text(
        "Mestrado Acadêmico em Ciência da Computação",
        font_size=20,
    )

    university = Text(
        "Universidade Federal do Ceará",
        font_size=20,
    )

    information = VGroup(
        author,
        program,
        university,
    ).arrange(
        DOWN,
        buff=0.2,
    )

    information.next_to(
        title_group,
        DOWN,
        buff=0.8,
    )

    self.play(
        FadeIn(
            information,
            shift=UP * 0.2,
        ),
        run_time=1.0,
    )

    orientador = Text("Orientador: Prof. Dr. Yuri Lenon Barbosa Nogueira", font_size=17)
    coorientador = Text("Coorientador: Prof. Dr. Joaquim Bento Cavalcante Neto", font_size=17)
    avaliador1 = Text("Examinador: Prof. Dr. Creto Augusto Vidal", font_size=17)
    avaliador2 = Text("Examinador: Prof. Dr. José Gilvan Rodrigues Maia", font_size=17)
    avaliador3 = Text("Examinador: Prof. Dr. Tarcísio Haroldo Cavalcante Pequeno", font_size=17)

    advisors = VGroup(
        orientador,
        coorientador,
        avaliador1,
        avaliador2,
        avaliador3
    ).arrange(
        direction=DOWN,
        buff=0.08,
        aligned_edge=LEFT,
    )

    advisors.next_to(information, DOWN, 0.8)

    self.play(
        FadeIn(advisors),
        run_time = 1.0,
    )

    self.next_slide()
