import streamlit as st
from video_utils import predict_video
import tempfile

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎭 Deepfake Video Detection System")
st.write("Upload a video and the model will classify it as REAL or FAKE.")

uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    if st.button("Analyze Video"):
        with st.spinner("Detecting..."):
            result = predict_video(tfile.name)
        st.success(result)
