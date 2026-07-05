import os
import cv2
import time
import torch
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from env.health_gathering import HealthGathering

def transform_image(img):
    # Move to CPU if needed
    img = img.detach().cpu()

    # Convert from (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0)       # (84, 84, 1 or 3)

    # If normalized, convert back to uint8
    if img.dtype != torch.uint8:
        img = (img * 255).clamp(0, 255).byte()

    img = img.numpy()

    # Convert RGB -> BGR (OpenCV expects BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img

def show_image(obs):
    image = transform_image(obs[0])
    scale = 6
    image_rgb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    image_bgr = image_rgb[:, :, ::-1]  # convert RGB to BGR for OpenCV

    cv2.imshow("game", image_bgr)
    cv2.waitKey(1)

def play_episode(env, agent, image_name:int):
    total_reward = 0
    finished = False
    info = {}
    obs, _ = env.reset(options={"image_name": image_name})
    ep_len = 0
    while not finished:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # show_image(obs_tensor)
            logits = agent(obs_tensor)
            action = torch.distributions.Categorical(logits=logits).sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ep_len += 1
        finished = terminated or truncated
        # time.sleep(0.1)
    return 4*ep_len, total_reward, info

def play_in_dir(dir:str, glaucoma_level:int):
    # checking if directory exists
    model_dir = f"{dir}/model"
    if not os.path.isdir(model_dir):
        return
    # creating envs
    num_to_envname = { 0:"random", 1:"square", 2:"circle", 3:"sin", 4:"grid" }
    envs = []
    for i in range(0, 5):
        info = {
            "glaucoma_level": glaucoma_level,
            "render_mode": None,
            "eval_layout": i
        }
        hg = HealthGathering(info, f"{dir}/{num_to_envname[i]}")
        envs.append(hg.make_env())
    # playing and generating images
    for model in os.listdir(model_dir):
        device = "cpu"
        image_name = int(model.split("_")[1].split(".")[0])
        agent = torch.load(f"{model_dir}/{model}", map_location=torch.device(device), weights_only=False)
        for env in envs:
            play_episode(env, agent, image_name)


# not paralellized version
# def generate_images(root_dir:str):
#     for dir1 in os.listdir(root_dir):
#         glaucoma_level = int(dir1.split("_")[0])
#         experiment = f"{root_dir}/{dir1}"
#         for dir2 in os.listdir(experiment):
#             subexperiment = f"{experiment}/{dir2}"
#             play_in_dir(subexperiment, glaucoma_level)

def _play_in_dir_job(args):
    subexperiment, glaucoma_level = args
    play_in_dir(subexperiment, glaucoma_level)

def generate_images(root_dir: str):
    jobs = []

    for dir1 in os.listdir(root_dir):
        glaucoma_level = int(dir1.split("_")[0])
        experiment = f"{root_dir}/{dir1}"

        for dir2 in os.listdir(experiment):
            subexperiment = f"{experiment}/{dir2}"
            jobs.append((subexperiment, glaucoma_level))

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        mp_context=ctx
    ) as executor:
        executor.map(_play_in_dir_job, jobs)

if __name__ == "__main__":
    generate_images("./logs_noreward")
