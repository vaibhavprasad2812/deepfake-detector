import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CNN FEATURE EXTRACTOR (ResNet18)
# -----------------------------
def build_cnn_feature_extractor():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model

# -----------------------------
# CNN + LSTM MODEL
# -----------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.cnn = build_cnn_feature_extractor()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.cnn(x)
        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1])

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = CNN_LSTM().to(device)
    model.load_state_dict(
        torch.load("model/deepfake_cnn_lstm.pth", map_location=device)
    )
    model.eval()
    return model

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FRAME SAMPLING
# -----------------------------
def sample_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return frames

    step = max(1, total // max_frames)
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# -----------------------------
# SIMPLE FACE EXTRACTION (center crop)
# -----------------------------
def extract_faces(frames):
    faces = []
    for frame in frames:
        h, w, _ = frame.shape
        size = min(h, w)
        cx, cy = w // 2, h // 2
        face = frame[
            cy - size // 4 : cy + size // 4,
            cx - size // 4 : cx + size // 4
        ]
        if face.size != 0:
            faces.append(face)
    return faces

# -----------------------------
# FACE → TENSOR
# -----------------------------
def faces_to_tensor(faces):
    tensors = [transform(face) for face in faces]
    return torch.stack(tensors)

# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_video(video_path):
    model = load_model()

    frames = sample_frames(video_path)
    faces = extract_faces(frames)

    if len(faces) == 0:
        return {
            "label": "No face detected",
            "confidence": 0,
            "faces": []
        }

    faces_tensor = faces_to_tensor(faces)
    faces_tensor = faces_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(faces_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "REAL" if prediction.item() == 0 else "FAKE"

    return {
        "label": label,
        "confidence": round(confidence.item() * 100, 2),
        "faces": faces
    }
