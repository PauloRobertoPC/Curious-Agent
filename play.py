import time
import cv2
import torch as th
from rllte.env import make_envpool_vizdoom_env, health_gathering
from rllte.env.utils import EnvPoolSync2Gymnasium, Gymnasium2Torch
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper

def transform_image(img):
    # Move to CPU if needed
    img = img.detach().cpu()

    # Convert from (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0)       # (84, 84, 1 or 3)

    # If normalized, convert back to uint8
    if img.dtype != th.uint8:
        img = (img * 255).clamp(0, 255).byte()

    img = img.numpy()

    # scalling the image
    scale = 6
    h, w = img.shape[:2]
    img = cv2.resize(
        img,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert RGB -> BGR (OpenCV expects BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def show_image(obs):
    img = transform_image(obs[0])
    cv2.imshow("game", img)
    cv2.waitKey(1)

device = "cuda"

agent = th.load("logs/health_gathering/2026-01-16-11-54-11/model/agent_102400.pth", map_location=th.device(device), weights_only=False)
agent.eval()

print(agent)

FPS = 10
FRAME_TIME = 1.0 / FPS

last = time.time()

tot = 0
ok = False
env = health_gathering(1, device, 1, False)
obs, info = env.reset()
while not ok:
    start = time.time()

    show_image(obs)

    with th.no_grad():
        logits = agent(obs)
        action = th.distributions.Categorical(logits=logits).sample()

    obs, reward, terminated, truncated, info = env.step(action)
    ok = terminated or truncated

    tot += 1
    elapsed = time.time() - start
    time.sleep(max(0, FRAME_TIME - elapsed))

print(tot*4)