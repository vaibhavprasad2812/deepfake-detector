import streamlit as st
import tempfile
from video_utils import predict_video

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="🎭",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎭 Deepfake Video Detection System</h1>
    <p style="text-align:center; font-size:18px;">
    Upload a video and the model will classify it as <b>REAL</b> or <b>FAKE</b>
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MP4 Video",
    type=["mp4", "mov", "avi"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Analyze Video"):
        progress = st.progress(0)
        status = st.empty()

        status.info("🔍 Extracting frames...")
        progress.progress(25)

        status.info("🙂 Detecting faces...")
        progress.progress(50)

        status.info("🧠 Running deep learning model...")
        result = predict_video(tfile.name)
        progress.progress(75)

        status.info("📊 Finalizing result...")
        progress.progress(100)

        st.markdown("---")

        # RESULT
        if result["label"] == "No face detected":
            st.error("❌ No face detected clearly in the video")
        else:
            if result["label"] == "REAL":
                st.success(f"✅ REAL ({result['confidence']}% confidence)")
            else:
                st.error(f"🚨 FAKE ({result['confidence']}% confidence)")

            st.markdown("### 👤 Extracted Face Samples")
            cols = st.columns(4)
            for i, face in enumerate(result["faces"][:4]):
                cols[i].image(face, use_container_width=True)
