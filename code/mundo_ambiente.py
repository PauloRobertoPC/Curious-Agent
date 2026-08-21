from manim import *
import numpy as np

def mundo_ambiente(self):
    self.remove(*self.mobjects)

    # ==========================================
    # PARTE 1: Recriar a cena passada (Transição invisível)
    # ==========================================
    cell_center = RIGHT * 2.0
    
    membrane = Circle(radius=1.7, color=YELLOW, stroke_width=4).move_to(cell_center)
    cytoplasm = Circle(radius=1.7, color=YELLOW, fill_color=GREEN_E, fill_opacity=0.3, stroke_width=0).move_to(cell_center)
    nucleus = Circle(radius=0.4, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.8).move_to(cell_center + RIGHT*0.2 + UP*0.2)
    nucleus_lbl = Text("Núcleo", font_size=14, color=WHITE).move_to(nucleus.get_center())
    
    mito1 = Ellipse(width=0.4, height=0.2, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + LEFT * 0.7 + UP * 0.3).rotate(PI/4)
    mito2 = Ellipse(width=0.3, height=0.15, color=ORANGE, fill_color=ORANGE, fill_opacity=0.8).move_to(cell_center + RIGHT * 0.5 + DOWN * 0.7).rotate(-PI/6)
    vac1 = Circle(radius=0.15, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + LEFT * 0.4 + DOWN * 0.6)
    vac2 = Circle(radius=0.1, color=PURPLE, fill_color=PURPLE, fill_opacity=0.7).move_to(cell_center + RIGHT * 0.8 + UP * 0.6)
    golgi_arcs = VGroup(Arc(radius=0.2, angle=PI/1.5, color=TEAL, stroke_width=4), Arc(radius=0.3, angle=PI/1.5, color=TEAL, stroke_width=4)).move_to(cell_center + LEFT * 0.2 + UP * 0.8).rotate(PI/4)
    organelles = VGroup(mito1, mito2, vac1, vac2, golgi_arcs)
    
    full_cell = VGroup(cytoplasm, membrane, nucleus, nucleus_lbl, organelles)
    cell_label = Text("Sistema Autopoiético", font_size=20, color=YELLOW).next_to(membrane, DOWN, buff=0.7)
    
    self.add(full_cell, cell_label)

    # Recriando os textos do slide passado
    old_title = Text("Produção de Sentido", font_size=40, weight=BOLD).to_edge(UP)
    old_subtitle = Text("(Sense-Making)", font_size=24, color=YELLOW).next_to(old_title, DOWN, buff=0.2)
    
    old_t1 = Text("• O mundo não é neutro", font_size=28, color=YELLOW, weight=BOLD)
    old_t1_sub = Text("A célula avalia o ambiente com base\nno que contribui ou não para a\nmanutenção da sua própria identidade.", font_size=22, color=LIGHT_GREY, line_spacing=1)
    old_group_t1 = VGroup(old_t1, old_t1_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.8).shift(UP * 0.5)

    old_t2 = Text("• O Significado Emerge", font_size=28, color=GREEN_C, weight=BOLD)
    old_t2_sub = Text("O que favorece a identidade\ntorna-se um recurso (atração).\nO que a ameaça torna-se\num perigo (repulsa).", font_size=22, color=LIGHT_GREY, line_spacing=1)
    old_group_t2 = VGroup(old_t2, old_t2_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(old_group_t1, DOWN, buff=0.6, aligned_edge=LEFT)

    old_good = Circle(radius=0.4, color=GREEN, fill_color=GREEN, fill_opacity=0.8).move_to(RIGHT * 5.0 + UP * 1.5)
    old_plus = Text("+", font_size=32, color=WHITE, weight=BOLD).move_to(old_good)
    old_good_group = VGroup(old_good, old_plus)
    old_bad = Triangle(color=RED, fill_color=RED, fill_opacity=0.8).scale(0.5).move_to(RIGHT * 5.0 + DOWN * 1.0)
    old_minus = Text("-", font_size=32, color=WHITE, weight=BOLD).move_to(old_bad)
    old_bad_group = VGroup(old_bad, old_minus)

    self.add(old_title, old_subtitle, old_group_t1, old_group_t2, old_good_group, old_bad_group)

    # ==========================================
    # PARTE 2: A Transição 
    # ==========================================
    title = Text("Ambiente vs. Mundo", font_size=40, weight=BOLD).to_edge(UP)
    
    self.play(
        FadeOut(old_title), FadeOut(old_subtitle), FadeOut(old_group_t1), FadeOut(old_group_t2),
        FadeOut(old_good_group), FadeOut(old_bad_group),
        Write(title)
    )
    self.next_slide()

    # ==========================================
    # PARTE 3: Reorganizar o Ecrã e criar o Observador
    # ==========================================
    new_center = LEFT * 4.0 + UP * 1.5
    new_label = Text("Bactéria\n(Metaboliza Açúcar)", font_size=20, color=GREEN_C)
    new_label.scale(0.65).next_to(new_center, DOWN, buff=1.3)
    
    self.play(
        full_cell.animate.scale(0.65).move_to(new_center),
        Transform(cell_label, new_label)
    )
    
    # Cria a célula "Observador"
    obs_center = LEFT * 4.0 + DOWN * 1.5
    obs_membrane = Circle(radius=1.7, color=BLUE, stroke_width=4).move_to(obs_center)
    obs_cyto = Circle(radius=1.7, color=BLUE, fill_color=GRAY, fill_opacity=0.3, stroke_width=0).move_to(obs_center)
    obs_nucleus = Circle(radius=0.4, color=PURPLE_E, fill_color=PURPLE_E, fill_opacity=0.8).move_to(obs_center + RIGHT*0.2 + UP*0.2)
    obs_nuc_lbl = Text("Núcleo", font_size=14, color=WHITE).move_to(obs_nucleus)
    obs_org1 = RegularPolygon(n=5, color=TEAL, fill_color=TEAL, fill_opacity=0.8).scale(0.3).move_to(obs_center + LEFT * 0.6 + UP * 0.4)
    obs_org2 = RegularPolygon(n=5, color=TEAL, fill_color=TEAL, fill_opacity=0.8).scale(0.2).move_to(obs_center + RIGHT * 0.5 + DOWN * 0.6)

    obs_full = VGroup(obs_cyto, obs_membrane, obs_nucleus, obs_nuc_lbl, obs_org1, obs_org2)
    obs_label = Text("Observador\n(Outra Identidade)", font_size=20, color=BLUE_C).next_to(obs_full, DOWN, buff=0.7)
    obs_group = VGroup(obs_full, obs_label).scale(0.65)

    self.play(FadeIn(obs_group, shift=UP*0.2))
    self.next_slide()

    # ==========================================
    # PARTE 4: A Síntese de Varela (Textos à Direita)
    # ==========================================
    p1 = Text("• Ambiente (Físico)", font_size=28, color=GRAY, weight=BOLD)
    p1_sub = Text("Apenas fluxos de matéria e energia.", font_size=22, color=LIGHT_GREY, line_spacing=1)

    p2 = Text("• Mundo (Cognitivo)", font_size=28, color=BLUE_C, weight=BOLD)
    p2_sub = Text("A relevância surge a partir da\nidentidade de cada organismo.", font_size=22, color=LIGHT_GREY, line_spacing=1)

    p3 = Text("• Exemplo (VARELA, 1991)", font_size=28, color=YELLOW, weight=BOLD)
    p3_sub = Text("Uma molécula de açúcar é energia\npara quem a metaboliza, mas\nirrelevante para outras identidades.", font_size=22, color=LIGHT_GREY, line_spacing=1)

    texts = VGroup(p1, p1_sub, p2, p2_sub, p3, p3_sub).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.2)

    self.play(FadeIn(p1), FadeIn(p1_sub))
    self.play(FadeIn(p2), FadeIn(p2_sub))
    self.play(FadeIn(p3), FadeIn(p3_sub))
    self.next_slide()

    # ==========================================
    # PARTE 5: A Molécula de Açúcar e as Diferentes Reações
    # ==========================================
    sugar_hex = RegularPolygon(n=6, color=WHITE, fill_color=WHITE, fill_opacity=0.3).scale(0.5).move_to(DOWN * 0.5 + RIGHT * 0.5)
    sugar_lbl = Text("Açúcar", font_size=18, color=WHITE).next_to(sugar_hex, DOWN, buff=0.2)
    sugar_group = VGroup(sugar_hex, sugar_lbl)

    self.play(FadeIn(sugar_group, scale=0.5))
    self.next_slide()

    line_bact = DashedLine(start=full_cell.get_right(), end=sugar_hex.get_top(), color=YELLOW, dash_length=0.1)
    self.play(Create(line_bact))
    
    self.play(
        line_bact.animate.set_color(GREEN),
        sugar_hex.animate.set_color(GREEN).set_fill(GREEN, opacity=0.8),
        Wiggle(full_cell, scale_value=1.05),
        run_time=1.5
    )
    self.next_slide()

    line_obs = DashedLine(start=obs_full.get_right(), end=sugar_hex.get_bottom(), color=YELLOW, dash_length=0.1)
    self.play(Create(line_obs))

    self.play(
        line_obs.animate.set_color(DARK_GRAY),
        obs_full.animate.set_opacity(0.5).set_opacity(1.0),
        run_time=1.5
    )
    
    obs_desc = Text("Irrelevante", font_size=16, color=GRAY).next_to(line_obs, DOWN, buff=0.1)
    bact_desc = Text("Energia (+)", font_size=16, color=GREEN).next_to(line_bact, UP, buff=0.1)
    
    self.play(Write(bact_desc), Write(obs_desc))
    self.next_slide()

    # ==========================================
    # PARTE 6: A Conclusão Enativa (Corpo, Mente e Ambiente)
    # ==========================================
    
    # Limpa os textos antigos de Varela
    self.play(FadeOut(texts))
    
    # Agrupa todo o diagrama visual criado nas partes anteriores
    diagram_group = VGroup(
        full_cell, cell_label, 
        obs_group, 
        sugar_group, 
        line_bact, bact_desc, 
        line_obs, obs_desc
    )
    
    # Move e minimiza o diagrama todo para a extrema esquerda
    self.play(
        diagram_group.animate.scale(0.7).to_edge(LEFT, buff=0.5),
        run_time=1.5
    )
    
    c1 = Text("O Corpo também importa!", font_size=32, color=RED_C, weight=BOLD)
    c2 = Text(
        "O significado das coisas não existe\n"
        "apenas de forma flutuante no ambiente.", 
        font_size=24, color=WHITE, line_spacing=1
    )
    c3 = Text(
        "A verdadeira cognição emerge da interação:", 
        font_size=24, color=LIGHT_GREY, line_spacing=1
    )
    
    triad = Text("Corpo + Mente + Ambiente", font_size=36, color=GREEN_C, weight=BOLD)
    
    # Alinha a conclusão e posiciona-a à direita do diagrama recuado
    conclusion_group = VGroup(c1, c2, c3, triad).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    conclusion_group.next_to(diagram_group, RIGHT, buff=1.0)
    
    self.play(FadeIn(c1, shift=LEFT*0.2))
    self.play(FadeIn(c2, shift=LEFT*0.2))
    self.next_slide()
    
    self.play(FadeIn(c3, shift=LEFT*0.2))
    self.play(Write(triad))
    self.play(Flash(triad, color=GREEN_C, line_length=0.4, num_lines=12))
    self.next_slide()

    # Limpa a tela inteira para o próximo slide
    self.play(*[FadeOut(mob) for mob in self.mobjects])
