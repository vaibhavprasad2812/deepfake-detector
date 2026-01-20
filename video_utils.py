import torch
import torch.nn as nn
import cv2
import numpy as np
from mtcnn import MTCNN
from torchvision import models, transforms
from PIL import Image

# --------------------------
# TRANSFORMS
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------
# MODEL DEFINITIONS
# --------------------------
def build_cnn_feature_extractor():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super(CNN_LSTM, self).__init__()
        self.cnn = build_cnn_feature_extractor()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch, seq, C, H, W = x.shape
        x = x.view(batch*seq, C, H, W)
        features = self.cnn(x)
        features = features.view(batch, seq, -1)
        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :]
        return self.classifier(out)


# --------------------------
# VIDEO PROCESSING FUNCTIONS
# --------------------------
def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(frame)
        i += 1

    cap.release()
    return frames

detector = MTCNN()

def extract_faces(frames):
    faces = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(rgb)
        if len(result) == 0:
            continue
        x, y, w, h = result[0]['box']
        x, y = max(0, x), max(0, y)
        face = rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        faces.append(face)
    return faces


def faces_to_tensor(faces):
    tensors = []
    for face in faces:
        img = Image.fromarray(face)
        img = transform(img)
        tensors.append(img)
    return torch.stack(tensors).unsqueeze(0)


# --------------------------
# MAIN PREDICTION FUNCTION
# --------------------------
def predict_video(video_path, model_path="model/deepfake_cnn_lstm.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = sample_frames(video_path)
    faces = extract_faces(frames)

    if len(faces) < 1:
        return "Error: No face found"

    faces_tensor = faces_to_tensor(faces).to(device)

    with torch.no_grad():
        output = model(faces_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    if pred == 0:
        return f"REAL (Confidence: {prob[0][0]:.4f})"
    else:
        return f"FAKE (Confidence: {prob[0][1]:.4f})"
