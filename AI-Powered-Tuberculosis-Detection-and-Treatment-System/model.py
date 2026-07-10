import torch
import torch.nn as nn
import timm

class TBXRayModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(TBXRayModel, self).__init__()

        self.convnext = timm.create_model(
            "convnextv2_large.fcmae",  # High-capacity ConvNeXt v2-Large
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.convnext(x)


# GradCAM module (optional, for inference and visualization)
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        target_logit = output[:, output.argmax()]
        target_logit.backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.size(0)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap)

        return heatmap, output
