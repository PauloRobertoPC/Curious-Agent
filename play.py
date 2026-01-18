import time
import torch
import numpy as np
import cv2
import torch as th
from rllte.env import health_gathering
from wrappers import image_wrapper, glaucoma_wrapper

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def instantiate_cam(model):
    target_layers = [
        # agent.encoder.trunk[-9], # Conv2d
        # agent.encoder.trunk[-8], # ReLU
        # agent.encoder.trunk[-7], # Conv2d
        # agent.encoder.trunk[-6], # ReLU
        agent.encoder.trunk[-5], # Conv2d
        # agent.encoder.trunk[-4], # ReLU
        # agent.encoder.trunk[-3], # Flatten
        # agent.encoder.trunk[-2], # Linear
        # agent.encoder.trunk[-1]  # ReLU
    ]

    # You can choose different CAM methods here
    cam = GradCAM(model=model, target_layers=target_layers)
    # cam = HiResCAM(model=model, target_layers=target_layers)
    # cam = ScoreCAM(model=model, target_layers=target_layers)
    # cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    # cam = AblationCAM(model=model, target_layers=target_layers)
    # cam = XGradCAM(model=model, target_layers=target_layers)
    # cam = EigenCAM(model=model, target_layers=target_layers)
    # cam = FullGrad(model=model, target_layers=target_layers)
    return cam


def transform_image(img):
    # Move to CPU if needed
    img = img.detach().cpu()

    # Convert from (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0)       # (84, 84, 1 or 3)

    # If normalized, convert back to uint8
    if img.dtype != th.uint8:
        img = (img * 255).clamp(0, 255).byte()

    img = img.numpy()

    # Convert RGB -> BGR (OpenCV expects BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img

def generate_cam(cam, img, targets):
    img = img.to("cpu")
    img_batch = img.to("cpu") #.to("cuda")
    grayscale_cam = cam(input_tensor=img_batch, targets=targets, aug_smooth=False, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # since we have a single image in the batch

    img_hwc = np.transpose(img.squeeze(0).numpy(), (1, 2, 0))/255.0
    rgb_image = np.repeat(img_hwc, 3, axis=2)
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return visualization

def show_image(obs, cam=None, targets=None):
    image = transform_image(obs[0])
    if cam is not None:
        cam_image = generate_cam(cam, obs, targets)
        image = np.concatenate((image, cam_image), axis=1)
    scale = 6
    image_rgb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    image_bgr = image_rgb[:, :, ::-1]  # convert RGB to BGR for OpenCV

    cv2.imshow("game", image_bgr)
    cv2.waitKey(1)

device = "cuda"

agent = th.load("logs/health_gathering_100g_rnd/2026-01-17-03-01-50/model/agent_1999872.pth", map_location=th.device(device), weights_only=False)
agent.eval()
cam = instantiate_cam(agent)

print(agent)

instantiate_cam(agent)
FPS = 5
FRAME_TIME = 1.0 / FPS

last = time.time()

tot = 0
ok = False
wrappers = [
    image_wrapper((84, 84)),
    glaucoma_wrapper(0, 100)
]
env = health_gathering(1, device, 1, False, wrappers)
obs, info = env.reset()
while not ok:
    start = time.time()


    with th.no_grad():
        logits = agent(obs)
        action = th.distributions.Categorical(logits=logits).sample()

    obs, reward, terminated, truncated, info = env.step(action)
    ok = terminated or truncated

    # show_image(obs)
    show_image(obs, cam, [ClassifierOutputTarget(action)])

    tot += 1
    elapsed = time.time() - start
    time.sleep(max(0, FRAME_TIME - elapsed))

print(tot*4)