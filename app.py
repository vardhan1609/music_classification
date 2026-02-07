import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os

# ---------------- CONFIG (MATCH TRAINING) ---------------
MODEL_PATH = "music_genre_model1005.h5"
ENCODER_PATH = "label_encoder50 (1).pkl"

SAMPLE_RATE = 22050
DURATION = 15                     # seconds
SAMPLES_PER_TRACK = 66150        # 22050 * 3

N_MELS = 128
MAX_LEN = 130                    # X_mel.shape[2]
# ---------------------------------------------------------

st.set_page_config(page_title="Audio Genre Classification", layout="centered")
st.title("🎵 Audio Genre Classification")
st.write("Upload **.mp3 or .wav** file(s)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, label_encoder = load_model_and_encoder()

# ---------------- UTILS ----------------
def pad_or_trim_wave(audio, max_len):
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    return audio

def pad_or_trim_2d(feature, max_len):
    if feature.shape[1] < max_len:
        feature = np.pad(feature, ((0,0),(0, max_len - feature.shape[1])))
    else:
        feature = feature[:, :max_len]
    return feature

# ---------------- FEATURE EXTRACTION ----------------
def extract_waveform(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio = pad_or_trim_wave(audio, SAMPLES_PER_TRACK)
    audio = audio.reshape(1, SAMPLES_PER_TRACK, 1)   # ✅ FIXED
    return audio

def extract_mel(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = pad_or_trim_2d(mel, MAX_LEN)
    mel = mel.reshape(1, N_MELS, MAX_LEN, 1)
    return mel

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload audio files",
    type=["mp3", "wav"],
    accept_multiple_files=True
)

# ---------------- PREDICTION ----------------
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")

    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"🎧 {uploaded_file.name}")

        temp_path = f"temp_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            wave = extract_waveform(temp_path)
            mel  = extract_mel(temp_path)

            preds = model.predict([wave, mel], verbose=0)

            confidence = float(np.max(preds) * 100)
            class_idx = int(np.argmax(preds))
            label = label_encoder.inverse_transform([class_idx])[0]

            st.write(f"**Predicted Genre:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2f}%`")
            st.progress(int(confidence))

        except Exception as e:
            st.error(f"Error processing file: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
