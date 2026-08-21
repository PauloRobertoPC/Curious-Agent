import os
import numpy as np
import cv2
import torch
from manim import *

def load_episode_data(data_path:str):
    filepath = data_path
    if not os.path.exists(filepath):
        print(f"THE FILE {filepath} DOES NOT EXIST")
        return [], [], {}
    
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    infos = data["infos"]
    infos = [info[0] for info in infos]
    infos = {
        key: [d[key] for d in infos]
        for key in infos[0]
    }
    return data["frames"].numpy(), data["rewards"].squeeze(-1).numpy(), infos


def grafico_rnd(self, data_path: str, image_indices: list[int], num_frames: int):
    # ==========================================
    # PARTE 1: Carregamento e Fatiamento de Dados
    # ==========================================
    frames, rewards, infos = load_episode_data(data_path)
    
    if len(frames) == 0:
        print("Erro: Dados não carregados.")
        return

    TOTAL_STEPS = min(num_frames, len(rewards))
    frames = frames[:TOTAL_STEPS]
    rewards = rewards[:TOTAL_STEPS]
    
    medikits_array = np.asarray(infos["MEDIKITS"])[:TOTAL_STEPS]
    medikit_timesteps = np.where(np.diff(medikits_array) > 0)[0] + 1
    image_indices = [idx for idx in image_indices if idx < TOTAL_STEPS]

    # ==========================================
    # PARTE 2: Layout - Imagem e Títulos
    # ==========================================
    title = Text("Recompensa Intrínseca e Condição Corporal", font_size=36, weight=BOLD).to_edge(UP, buff=0.2)
    self.play(Write(title))

    first_frame = frames[0]
    if first_frame.shape[0] == 3: # Se for [C, H, W], transpor para [H, W, C]
        first_frame = np.transpose(first_frame, (1, 2, 0))
    
    # CORREÇÃO: Fatiando a imagem para pegar apenas a metade esquerda (visão pura 84x84)
    first_frame = first_frame[:, :84]
    
    if first_frame.dtype != np.uint8:
        first_frame = (first_frame * 255).astype(np.uint8) if first_frame.max() <= 1.0 else first_frame.astype(np.uint8)

    # Imagem agora quadrada (84x84 redimensionada)
    obs_image = ImageMobject(first_frame).scale_to_fit_height(2.5)
    obs_image.to_corner(UL, buff=1.0).shift(DOWN * 0.5 + RIGHT * 1.5)
    
    img_bg = SurroundingRectangle(obs_image, color=WHITE, buff=0.05, stroke_width=2)
    img_lbl = Text("Visão do Agente", font_size=20).next_to(img_bg, UP, buff=0.1)
    
    img_group = Group(img_bg, obs_image, img_lbl)
    self.play(FadeIn(img_group))

    # Textos Dinâmicos ao lado da imagem
    step_lbl = Text("Passo: 0", font_size=24, color=YELLOW)
    reward_lbl = Text("Recompensa: 0.000", font_size=24, color=BLUE_C)
    status_group = VGroup(step_lbl, reward_lbl).arrange(DOWN, aligned_edge=LEFT).next_to(img_bg, RIGHT, buff=1.0)
    
    self.play(FadeIn(status_group))

    # ==========================================
    # PARTE 3: O Gráfico (Axes)
    # ==========================================
    y_max = max(rewards) * 1.2 if max(rewards) > 0 else 0.05
    y_step = y_max / 5 if y_max > 0 else 0.01
    
    # Eixos levemente reduzidos na largura para não esmagar o texto Y na esquerda
    ax = Axes(
        x_range=[0, TOTAL_STEPS, max(1, TOTAL_STEPS // 10)],
        y_range=[0, y_max, y_step],
        x_length=10.5,
        y_length=3.0,
        axis_config={"font_size": 20}
    ).to_edge(DOWN, buff=0.6).to_edge(RIGHT, buff=0.5)

    # CORREÇÃO DOS LABELS PARA EVITAR OVERLAP:
    # Coloca "Steps" logo abaixo da seta do eixo X
    x_lbl = Text("Steps", font_size=20, slant=ITALIC).next_to(ax.x_axis.get_end(), DOWN, buff=0.2)
    # Coloca "Intrinsic Reward" seguramente à esquerda do eixo Y
    y_lbl = Text("Intrinsic Reward", font_size=20, slant=ITALIC).rotate(PI/2).next_to(ax.y_axis, LEFT, buff=0.3)
    
    self.play(Create(ax), Write(x_lbl), Write(y_lbl))

    # ==========================================
    # PARTE 4: Updaters Otimizados (Renderização Rápida)
    # ==========================================
    tracker = ValueTracker(0)
    all_pts = [ax.c2p(i, rewards[i]) for i in range(TOTAL_STEPS)]
    
    curve = VMobject(color=BLUE)
    curve.last_t = -1 
    
    def update_curve(mob):
        t = int(tracker.get_value())
        if t != mob.last_t and t > 0:
            mob.last_t = t
            t_safe = min(t, TOTAL_STEPS - 1)
            mob.set_points_as_corners(all_pts[:t_safe+1])
            
    curve.add_updater(update_curve)
    self.add(curve)

    obs_image.last_t = -1
    def update_image(mob):
        t = int(tracker.get_value())
        if t != mob.last_t and t < TOTAL_STEPS:
            mob.last_t = t
            frame = frames[t]
            if frame.shape[0] == 3: 
                frame = np.transpose(frame, (1, 2, 0))
                
            # CORREÇÃO: Fatiando no loop para garantir que só a metade esquerda apareça
            frame = frame[:, :84]
            frame = np.ascontiguousarray(frame)
            
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            mob.pixel_array = rgba
            
    obs_image.add_updater(update_image)

    status_group.last_t = -1
    def update_texts(mob):
        t = int(tracker.get_value())
        if t != getattr(mob, "last_t", -1) and t < TOTAL_STEPS:
            mob.last_t = t
            mob[0].become(Text(f"Passo: {t}", font_size=24, color=YELLOW).move_to(mob[0], aligned_edge=LEFT))
            mob[1].become(Text(f"Recompensa: {rewards[t]:.4f}", font_size=24, color=BLUE_C).move_to(mob[1], aligned_edge=LEFT))
            
    status_group.add_updater(update_texts)
    self.next_slide()

    # ==========================================
    # PARTE 5: A Animação em Segmentos
    # ==========================================
    
    events = [{"time": mt, "type": "medikit"} for mt in medikit_timesteps]
    events += [{"time": st, "type": "stop"} for st in image_indices]
    events.sort(key=lambda x: x["time"])
    
    current_t = 0
    stop_counter = 1
    
    SEGUNDOS_POR_PASSO = 0.08 
    
    for event in events:
        target_t = event["time"]
        if target_t <= current_t or target_t >= TOTAL_STEPS:
            continue
            
        duration = (target_t - current_t) * SEGUNDOS_POR_PASSO 
        self.play(tracker.animate.set_value(target_t), run_time=duration, rate_func=linear)
        
        current_t = target_t
        pt_coord = all_pts[current_t]
        
        if event["type"] == "medikit":
            star = Star(color=GREEN, fill_color=GREEN, fill_opacity=1).scale(0.15).move_to(pt_coord)
            self.play(
                FadeIn(star, shift=UP*0.2), 
                Flash(img_bg, color=GREEN, line_length=0.4),
                run_time=0.5
            )
            
        elif event["type"] == "stop":
            dot = Dot(color=RED, radius=0.08).move_to(pt_coord)
            lbl = Text(str(stop_counter), font_size=18, color=RED, weight=BOLD).next_to(dot, RIGHT, buff=0.1)
            img_num = Text(str(stop_counter), font_size=40, color=RED, weight=BOLD).move_to(obs_image)
            
            self.play(FadeIn(dot), Write(lbl), FadeIn(img_num))
            self.next_slide() 
            self.play(FadeOut(img_num)) 
            stop_counter += 1

    if current_t < TOTAL_STEPS - 1:
        duration = (TOTAL_STEPS - 1 - current_t) * SEGUNDOS_POR_PASSO
        self.play(tracker.animate.set_value(TOTAL_STEPS - 1), run_time=duration, rate_func=linear)

    self.next_slide()
    self.play(*[FadeOut(mob) for mob in self.mobjects])
