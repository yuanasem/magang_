import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TBXRayModel
from med_recommendation import MedicationRecommender
import os

def get_inference_transforms(img_size=(384, 384)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def infer_and_visualize(image_path, model_path="best_model_updated.pth", diag_thresh=[0.5, 0.75]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TBXRayModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("L")
    tensor = get_inference_transforms()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, 1)[0].cpu().numpy()

    idx = np.argmax(probs)
    classes = ["Healthy", "Sick", "TB"]
    diag = classes[idx]
    conf = probs[idx] * 100

    if diag == "TB":
        sever = "Severe" if conf >= diag_thresh[1]*100 else "Moderate" if conf >= diag_thresh[0]*100 else "Mild"
    elif diag == "Sick":
        sever = "Moderate" if conf >= diag_thresh[0]*100 else "Mild"
    else:
        sever = "None"

    try:
        rec = MedicationRecommender()
        meds = rec.recommend_medication(f"Diagnosis: {diag}, Severity: {sever}, Confidence: {conf:.2f}%")
    except Exception as e:
        meds = f"⚠️ Medication recommendation failed: {e}"

    disclaimer = "Disclaimer: For research only. Not medical advice."

    return diag, sever, conf, meds, disclaimer

