# main.py
import streamlit as st
from pathlib import Path
from app.inference import process_video

st.set_page_config(page_title="AI Video Threat Classifier", layout="wide")

def main():
    st.title("AI Video Threat Classifier")
    st.write("Carica un video: l’output mostrerà solo il video annotato con l’equipaggiamento e 3 barre (High/Medium/Low).")

    uploaded = st.file_uploader("Seleziona un video", type=["mp4", "avi", "mov", "mpeg4"])

    if uploaded:
        # salva input su disco
        in_path  = Path("input_video.mp4")
        out_path = Path("output_video.mp4")
        with open(in_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Elaborazione in corso..."):
            # puoi regolare every_nth_frame=2 per velocizzare
            process_video(str(in_path), str(out_path), every_nth_frame=1, smooth=5)

        st.success("Fatto!")
        st.video(str(out_path))

if __name__ == "__main__":
    main()
