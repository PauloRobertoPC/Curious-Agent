import cv2
import torch
import numpy as np

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ModelCamWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ModelCamWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.extract_features(x)
        if self.model.share_features_extractor:
            latent_pi, latent_vf = self.model.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.model.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.model.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        distribution = self.model._get_action_dist_from_latent(latent_pi)
        return distribution.distribution.logits

def instantiate_cam(model):
    target_layers = [
        # model.model.features_extractor.cnn[-7],
        # model.model.features_extractor.cnn[-6],
        # model.model.features_extractor.cnn[-5],
        # model.model.features_extractor.cnn[-4],
        model.model.features_extractor.cnn[-3],
        # model.model.features_extractor.cnn[-2],
        # model.model.features_extractor.cnn[-1], DON't USE THIS ONE
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

# @torch.enable_grad()
def generate_cam(cam, img, targets):
    img_batch = torch.from_numpy(img).unsqueeze(0) #.to("cuda")
    grayscale_cam = cam(input_tensor=img_batch, targets=targets, aug_smooth=False, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # since we have a single image in the batch

    img_hwc = np.transpose(img, (1, 2, 0))
    rgb_image = np.repeat(img_hwc, 3, axis=2)
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return visualization

def show_game(obs, cam=None, targets=None):
    img_hwc = np.transpose(obs, (1, 2, 0))
    image = (np.repeat(img_hwc, 3, axis=2)*255).astype(np.uint8)
    if cam is not None:
        cam_image = generate_cam(cam, obs, targets)
        image = np.concatenate((image, cam_image), axis=1)
    image_rgb = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    image_bgr = image_rgb[:, :, ::-1]  # convert RGB to BGR for OpenCV
    cv2.imshow("Agent Playing", image_bgr)
    cv2.waitKey(1)
