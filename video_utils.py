import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import cv2
import numpy as np

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CNN FEATURE EXTRACTOR
# -----------------------------
def build_cnn_feature_extractor():
    model = models.resnet18(weights=None)   # IMPORTANT: no pre-trained weights for deployment
    model.fc = nn.Identity()                # Output = 512 features
    return model

# -----------------------------
# CNN + LSTM MODEL
# -----------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super(CNN_LSTM, self).__init__()

        self.cnn = build_cnn_feature_extractor()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x = (batch, 16, 3, 224, 224)
        batch, seq, C, H, W = x.shape

        x = x.view(batch * seq, C, H, W)
        features = self.cnn(x)              # → (batch*16, 512)
        features = features.view(batch, seq, -1)

        lstm_out, _ = self.lstm(features)   # → (batch, 16, hidden)
        output = lstm_out[:, -1, :]         # Last timestep

        return self.classifier(output)      # → (batch, 2)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
def load_model(model_path="model/deepfake_cnn_lstm.pth"):
    model = CNN_LSTM().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

# -----------------------------
# FRAME SAMPLING
# -----------------------------
def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return []

    frame_ids = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

# -----------------------------
# FACE EXTRACTION USING MTCNN
# -----------------------------
mtcnn = MTCNN(image_size=224, margin=20, device=device)

def extract_face(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)

    if face is None:
        return None

    return face

# -----------------------------
# MAIN PREDICT FUNCTION
# -----------------------------
def predict_video(video_path):
    model = load_model()

    frames = sample_frames(video_path, num_frames=16)

    if len(frames) == 0:
        return "Error: Could not read frames"

    faces = []
    for frame in frames:
        f = extract_face(frame)
        if f is not None:
            faces.append(f)

    if len(faces) < 4:
        return "Error: Face not detected clearly"

    faces_tensor = torch.stack(faces).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(faces_tensor)
        pred = torch.argmax(output, dim=1).item()

    return "FAKE" if pred == 1 else "REAL"
