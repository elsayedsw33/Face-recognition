import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# =========================
# Paths (model files must sit next to this file)
# =========================

BASE_DIR = Path(__file__).resolve().parent

YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = BASE_DIR / "face_recognition_sface_2021dec.onnx"
SPOOF_PATH = BASE_DIR / "best_model.onnx"
DATABASE_PATH = BASE_DIR / "face_database.npy"
THRESHOLD_PATH = BASE_DIR / "best_threshold.npy"

REQUIRED_FILES = [
    YUNET_PATH,
    SFACE_PATH,
    SPOOF_PATH,
    DATABASE_PATH,
    THRESHOLD_PATH,
]

# The YuNet detector is shared between browser sessions (cached resource) and
# its input size is changed per image, so inference is serialized with a lock.
_inference_lock = threading.Lock()


# =========================
# Load models (once)
# =========================

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    face_database = np.load(
        str(DATABASE_PATH),
        allow_pickle=True
    ).item()

    threshold = float(
        np.load(str(THRESHOLD_PATH))
    )

    detector = cv2.FaceDetectorYN_create(
        str(YUNET_PATH),
        "",
        (320, 320),
        score_threshold=0.60,
        nms_threshold=0.30,
        top_k=5000
    )

    recognizer = cv2.FaceRecognizerSF_create(
        str(SFACE_PATH),
        ""
    )

    # Anti-spoofing ONNX model (ONNX Runtime). Loaded once; only used by Live Camera.
    spoof_sess = ort.InferenceSession(
        str(SPOOF_PATH),
        providers=["CPUExecutionProvider"]
    )
    spoof_input_name = spoof_sess.get_inputs()[0].name

    return {
        "database": face_database,
        "threshold": threshold,
        "detector": detector,
        "recognizer": recognizer,
        "spoof_sess": spoof_sess,
        "spoof_input_name": spoof_input_name,
    }


# =========================
# Recognition (unchanged logic)
# =========================

def recognize_embedding(
    embedding,
    database,
    threshold
):

    best_name = "Unknown"
    best_score = -1.0

    for name, person_embeddings in database.items():

        scores = person_embeddings @ embedding
        score = float(np.max(scores))

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        best_name = "Unknown"

    return best_name, best_score


# =========================
# Anti-spoofing (unchanged preprocessing / class interpretation)
# =========================

def run_anti_spoofing(face_crop, models):
    """Return True if the (expanded) face crop is classified as a real face."""

    # Resize and convert to float32
    resized_crop = cv2.resize(face_crop, (128, 128))
    img_float = resized_crop.astype(np.float32) / 255.0

    # Convert BGR (OpenCV default) to RGB (PyTorch default)
    img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

    # Apply ImageNet Mean and Standard Deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_rgb - mean) / std

    # (Height, Width, Channels) -> (Channels, Height, Width)
    blob = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension
    blob = np.expand_dims(blob, axis=0)

    spoof_preds = models["spoof_sess"].run(
        None,
        {models["spoof_input_name"]: blob}
    )[0]

    scores = spoof_preds[0] if len(spoof_preds.shape) > 1 else spoof_preds
    predicted_class = np.argmax(scores)

    # IMPORTANT: Change to `== 0` if the model marks real faces as 0 instead of 1
    return bool(predicted_class == 1)


# =========================
# Face recognition (align -> SFace embedding -> normalize -> database match)
# =========================

def run_face_recognition(frame, face, models):
    """Return (name, score), or None if the embedding is invalid."""

    rec = models["recognizer"]

    # Align
    aligned = rec.alignCrop(
        frame,
        face
    )

    # SFace embedding
    embedding = rec.feature(
        aligned
    ).flatten()

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    embedding = embedding / norm

    # Recognize
    return recognize_embedding(
        embedding,
        models["database"],
        models["threshold"]
    )


# =========================
# Single face prediction
# =========================

def predict_face(frame, face, models, use_anti_spoofing=True):
    """Predict one detected face.

    use_anti_spoofing=True  (Live Camera):
        anti-spoofing -> if real: SFace recognition
    use_anti_spoofing=False (Upload Photo):
        SFace recognition only; anti-spoofing is never called

    Returns a result dict, or None if the face has to be skipped
    (empty crop or zero-norm embedding), same as the original script.
    """

    h, w = frame.shape[:2]
    x, y, fw, fh = face[:4].astype(int)
    box = (int(x), int(y), int(fw), int(fh))

    if use_anti_spoofing:

        # Expand bounding box (20% margin) for the anti-spoofing crop
        margin_w = int(fw * 0.2)
        margin_h = int(fh * 0.2)

        # Keep crop boundaries inside the frame
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(w, x + fw + margin_w)
        y2 = min(h, y + fh + margin_h)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        if not run_anti_spoofing(face_crop, models):
            return {
                "name": None,
                "score": None,
                "is_real": False,
                "status": "Spoof Detected",
                "box": box,
            }

        is_real = True
        status = "Real"

    else:
        # Photo mode: no anti-spoofing check was performed
        is_real = None
        status = "Photo Prediction"

    recognition = run_face_recognition(frame, face, models)

    if recognition is None:
        return None

    name, score = recognition

    return {
        "name": name,
        "score": score,
        "is_real": is_real,
        "status": status,
        "box": box,
    }


# =========================
# Image prediction (used by BOTH modes)
# =========================

def predict_image(image_bgr, models, use_anti_spoofing=True):
    """Detect all faces in a BGR image and return a list of result dicts."""

    h, w = image_bgr.shape[:2]

    with _inference_lock:

        detector = models["detector"]
        detector.setInputSize((w, h))

        _, faces = detector.detect(image_bgr)

        results = []

        if faces is not None:
            for face in faces:
                result = predict_face(
                    image_bgr,
                    face,
                    models,
                    use_anti_spoofing=use_anti_spoofing
                )
                if result is not None:
                    results.append(result)

    # Left-to-right order so "Face 1, Face 2..." matches the picture
    results.sort(key=lambda r: r["box"][0])

    return results


# =========================
# Helpers / UI
# =========================

def decode_image(uploaded_file):
    """Convert a Streamlit uploaded file to a BGR OpenCV image."""

    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def show_results(results):
    """Display predictions with normal Streamlit elements (no image overlays)."""

    st.subheader("Prediction")

    if not results:
        st.warning("No face detected in the image.")
        return

    for i, result in enumerate(results, start=1):

        if len(results) > 1:
            st.markdown(f"#### Face {i}")

        # Only the anti-spoofing path can produce this (Live Camera)
        if result["is_real"] is False:
            st.error("Status: Spoof Detected")
            continue

        if result["name"] == "Unknown":
            st.warning("Face not recognized (Unknown).")
        else:
            st.success(f"Recognized: {result['name']}")

        col_name, col_score, col_status = st.columns(3)
        col_name.metric("Name", result["name"])
        col_score.metric("Similarity", f"{result['score']:.2f}")
        col_status.metric("Status", result["status"])


def run_and_show(image_bgr, models):
    """Upload Photo: recognition only, anti-spoofing is NOT run."""

    with st.spinner("Analyzing..."):
        results = predict_image(image_bgr, models, use_anti_spoofing=False)
    show_results(results)


@st.cache_resource(ttl=3600)
def get_ice_servers():
    """ICE servers for WebRTC. Needed when the app is deployed (not on localhost).

    - STUN (free, Google) is always included.
    - TURN relay is added if credentials are found in Streamlit secrets:
        * Twilio:  TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN   (needs `twilio` package)
        * Generic: TURN_URLS (list) + TURN_USERNAME + TURN_CREDENTIAL
    """

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
            from twilio.rest import Client

            client = Client(
                st.secrets["TWILIO_ACCOUNT_SID"],
                st.secrets["TWILIO_AUTH_TOKEN"]
            )
            return client.tokens.create().ice_servers

        if "TURN_URLS" in st.secrets:
            ice_servers.append({
                "urls": list(st.secrets["TURN_URLS"]),
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            })
    except Exception:
        # No secrets configured (e.g. running locally) or TURN setup failed:
        # fall back to STUN only.
        pass

    return ice_servers


def run_live_camera(models):
    """Live Camera: anti-spoofing + recognition. The video stays clean,
    predictions update below it."""

    # Latest frame from the browser camera, shared with the video callback thread.
    if "live_holder" not in st.session_state:
        st.session_state.live_holder = {
            "frame": None,
            "count": 0,
            "lock": threading.Lock(),
        }
    holder = st.session_state.live_holder

    def video_frame_callback(frame):
        # Only store the frame (cheap) and return it untouched: no overlays.
        image_bgr = frame.to_ndarray(format="bgr24")
        with holder["lock"]:
            holder["frame"] = image_bgr
            holder["count"] += 1
        return frame

    ctx = webrtc_streamer(
        key="live-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        rtc_configuration={"iceServers": get_ice_servers()},
        async_processing=True,
    )

    st.caption("Click START, allow camera access, and predictions update live below the video.")

    result_box = st.empty()
    last_count = -1

    # Prediction loop: always works on the newest frame, skipping any it can't keep up with.
    while ctx.state.playing:

        with holder["lock"]:
            count = holder["count"]
            image_bgr = holder["frame"]

        if image_bgr is None:
            result_box.info("Waiting for the camera...")
            time.sleep(0.1)
            continue

        if count == last_count:
            time.sleep(0.03)
            continue

        last_count = count

        results = predict_image(image_bgr, models, use_anti_spoofing=True)

        with result_box.container():
            show_results(results)


def main():
    st.set_page_config(
        page_title="Face Recognition & Anti-Spoofing",
        layout="centered"
    )

    st.title("Face Recognition & Anti-Spoofing")

    missing = [p.name for p in REQUIRED_FILES if not p.exists()]
    if missing:
        st.error(
            "Missing required file(s) next to app.py: " + ", ".join(missing)
        )
        st.stop()

    models = load_models()

    mode = st.radio(
        "Mode",
        ["Upload Photo", "Live Camera"],
        horizontal=True
    )

    if mode == "Upload Photo":

        uploaded = st.file_uploader(
            "Upload a photo",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:
            image_bgr = decode_image(uploaded)

            if image_bgr is None:
                st.error("Could not read this image. Try another file.")
                return

            st.image(image_bgr, channels="BGR", use_container_width=True)
            run_and_show(image_bgr, models)

    else:

        run_live_camera(models)


if __name__ == "__main__":
    main()
