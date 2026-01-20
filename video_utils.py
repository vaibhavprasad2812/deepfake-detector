import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# CNN + LSTM MODEL
# -----------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super(CNN_LSTM, self).__init__()

        base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base_cnn.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        features = self.cnn(x)
        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :]
        return self.fc(out)


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(model_path="model/deepfake_cnn_lstm.pth"):
    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -----------------------------
# FACE DETECTOR (NO TENSORFLOW)
# -----------------------------
mtcnn = MTCNN(keep_all=True, device=device)


# -----------------------------
# FRAME SAMPLING
# -----------------------------
def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
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


# -----------------------------
# FACE EXTRACTION
# -----------------------------
def extract_faces(frames):
    faces = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is None:
            continue

        for box in boxes[:1]:  # take first face only
            x1, y1, x2, y2 = map(int, box)
            face = rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_img = Image.fromarray(face)
            face_tensor = transform(face_img)
            faces.append(face_tensor)

    return faces


# -----------------------------
# VIDEO TO TENSOR
# -----------------------------
def faces_to_tensor(faces):
    if len(faces) == 0:
        return None
    faces = torch.stack(faces)
    faces = faces.unsqueeze(0)  # (1, T, C, H, W)
    return faces.to(device)


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_video(video_path):
    model = load_model()

    frames = sample_frames(video_path)
    if len(frames) == 0:
        return "Error: Could not read video"

    faces = extract_faces(frames)
    if len(faces) < 3:
        return "Error: Face not detected clearly"

    input_tensor = faces_to_tensor(faces)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    if pred == 0:
        return f"REAL (Confidence: {probs[0][0]:.2f})"
    else:
        return f"FAKE (Confidence: {probs[0][1]:.2f})"
