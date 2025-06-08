import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x) * x
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.sa(sa_input)
        return ca * sa

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.layer1[0].add_module("cbam", CBAMBlock(256))
        self.model.layer2[0].add_module("cbam", CBAMBlock(512))
        self.model.layer3[0].add_module("cbam", CBAMBlock(1024))
        self.model.layer4[0].add_module("cbam", CBAMBlock(2048))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class_dict = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy", 4: "Background_without_leaves", 5: "Blueberry___healthy",
    6: "Cherry___Powdery_mildew", 7: "Cherry___healthy", 8: "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    9: "Corn___Common_rust", 10: "Corn___Northern_Leaf_Blight", 11: "Corn___healthy",
    12: "Grape___Black_rot", 13: "Grape___Esca_(Black_Measles)", 14: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    15: "Grape___healthy", 16: "Orange___Haunglongbing_(Citrus_greening)", 17: "Peach___Bacterial_spot",
    18: "Peach___healthy", 19: "Pepper,_bell___Bacterial_spot", 20: "Pepper,_bell___healthy",
    21: "Potato___Early_blight", 22: "Potato___Late_blight", 23: "Potato___healthy",
    24: "Raspberry___healthy", 25: "Soybean___healthy", 26: "Squash___Powdery_mildew",
    27: "Strawberry___Leaf_scorch", 28: "Strawberry___healthy", 29: "Tomato___Bacterial_spot",
    30: "Tomato___Early_blight", 31: "Tomato___Late_blight", 32: "Tomato___Leaf_Mold",
    33: "Tomato___Septoria_leaf_spot", 34: "Tomato___Spider_mites Two-spotted_spider_mite",
    35: "Tomato___Target_Spot", 36: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    37: "Tomato___Tomato_mosaic_virus", 38: "Tomato___healthy"
}


def load_model():
    model = ResNet50_CBAM(num_classes=39).to(device)

    state_dict = torch.load("model/model.pth", map_location=device)

    # Only load matching keys
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered_dict)
    model.load_state_dict(model_state)

    model.eval()
    return model

def generate_cam_image(model, image_path, class_dict):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    features_blobs = []

    def hook_fn(module, input, output):
        features_blobs.append(output.detach().cpu())

    final_conv = model.model.layer4[-1].conv3
    hook_handle = final_conv.register_forward_hook(hook_fn)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    hook_handle.remove()

    fc_weights = model.model.fc.weight.detach().cpu().numpy()
    feature_conv = features_blobs[0][0].numpy()
    bz, nc, h, w = features_blobs[0].shape

    cam = np.dot(fc_weights[pred_class], feature_conv.reshape(nc, h * w)).reshape(h, w)
    cam -= np.min(cam)
    cam /= np.max(cam) + 1e-5
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cv2.resize(cam, (512, 512)), cv2.COLORMAP_JET)

    orig = cv2.cvtColor(np.array(image.resize((512, 512))), cv2.COLOR_RGB2BGR)
    result = cv2.addWeighted(heatmap, 0.5, orig, 0.5, 0)

    result_path = os.path.join("static/results", os.path.basename(image_path))
    cv2.imwrite(result_path, result)

    return pred_class, class_dict[pred_class], result_path

