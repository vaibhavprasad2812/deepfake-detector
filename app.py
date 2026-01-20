# import streamlit as st
# from video_utils import predict_video
# import tempfile

# st.set_page_config(page_title="Deepfake Detector", layout="centered")

# st.title("🎭 Deepfake Video Detection System")
# st.write("Upload a video and the model will classify it as REAL or FAKE.")

# uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4", "mov", "avi"])

# if uploaded_file is not None:
#     # Save temporary file
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())

#     st.video(tfile.name)

#     if st.button("Analyze Video"):
#         with st.spinner("Detecting..."):
#             result = predict_video(tfile.name)
#         st.success(result)


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
st.markdown("""
<h1 style="text-align:center;">🎭 Deepfake Video Detection System</h1>
<p style="text-align:center; font-size:18px;">
Upload a video and detect whether it is <b>REAL</b> or <b>FAKE</b>
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MP4 video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Analyze Video"):
        progress = st.progress(0)
        status = st.empty()

        status.info("📽 Extracting frames...")
        progress.progress(25)

        status.info("🙂 Extracting faces...")
        progress.progress(50)

        status.info("🧠 Running deep learning model...")
        result = predict_video(temp.name)
        progress.progress(75)

        status.info("📊 Finalizing result...")
        progress.progress(100)

        st.markdown("---")

        if result["label"] == "No face detected":
            st.error("❌ No face detected in video")
        else:
            if result["label"] == "REAL":
                st.success(f"✅ REAL ({result['confidence']}%)")
            else:
                st.error(f"🚨 FAKE ({result['confidence']}%)")

            st.markdown("### 👤 Extracted Face Samples")
            cols = st.columns(4)
            for i, face in enumerate(result["faces"][:4]):
                cols[i].image(face, use_container_width=True)
