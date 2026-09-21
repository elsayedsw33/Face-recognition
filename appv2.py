import base64
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
import streamlit.components.v1 as components

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


# =========================
# Live camera (browser camera -> frames over Streamlit's own connection)
# No WebRTC, so no STUN/TURN servers are needed, locally or deployed.
# =========================

LIVE_INTERVAL_MS = 500     # how often the browser sends one frame to the app
LIVE_MAX_WIDTH = 640       # frames are downscaled to this width before sending
LIVE_IDLE_TIMEOUT = 20     # background worker stops after this many seconds without frames

LIVE_CAM_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; font-family: "Source Sans Pro", sans-serif; }
  #wrap { display: flex; flex-direction: column; gap: 8px; }
  button { padding: 8px 16px; font-size: 15px; border-radius: 8px; border: 1px solid #bbb;
           background: #fff; color: #222; cursor: pointer; align-self: flex-start; }
  button:hover { border-color: #ff4b4b; color: #ff4b4b; }
  video { width: 100%; border-radius: 8px; background: #000; display: none; }
  #msg { font-size: 14px; color: #888; }
</style>
</head>
<body>
<div id="wrap">
  <button id="toggle">Start camera</button>
  <video id="video" autoplay playsinline muted></video>
  <div id="msg"></div>
</div>
<canvas id="canvas" style="display:none"></canvas>
<script>
  function post(type, data) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data || {}), "*");
  }
  function setHeight() {
    post("streamlit:setFrameHeight", {height: document.getElementById("wrap").offsetHeight + 4});
  }
  function sendValue(value) {
    post("streamlit:setComponentValue", {value: value, dataType: "json"});
  }

  var intervalMs = 500;
  var maxWidth = 640;

  window.addEventListener("message", function (e) {
    if (e.data && e.data.type === "streamlit:render") {
      var a = e.data.args || {};
      intervalMs = a.interval_ms || intervalMs;
      maxWidth = a.max_width || maxWidth;
    }
  });

  var btn = document.getElementById("toggle");
  var video = document.getElementById("video");
  var msg = document.getElementById("msg");
  var canvas = document.getElementById("canvas");
  var stream = null;
  var timer = null;

  function captureFrame() {
    if (!video.videoWidth) { return; }
    var scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    sendValue({id: Date.now(), data: canvas.toDataURL("image/jpeg", 0.8)});
  }

  async function start() {
    msg.textContent = "";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      msg.textContent = "Camera access needs HTTPS (or localhost).";
      setHeight();
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: "user", width: {ideal: 1280}, height: {ideal: 720}},
        audio: false
      });
    } catch (err) {
      msg.textContent = "Could not open the camera: " + err.message;
      setHeight();
      return;
    }
    video.srcObject = stream;
    video.style.display = "block";
    btn.textContent = "Stop camera";
    timer = setInterval(captureFrame, intervalMs);
    setHeight();
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); }
    stream = null;
    video.srcObject = null;
    video.style.display = "none";
    btn.textContent = "Start camera";
    sendValue(null);
    setHeight();
  }

  btn.addEventListener("click", function () { if (stream) { stop(); } else { start(); } });
  video.addEventListener("loadedmetadata", setHeight);
  new ResizeObserver(setHeight).observe(document.getElementById("wrap"));

  post("streamlit:componentReady", {apiVersion: 1});
  setHeight();
</script>
</body>
</html>
"""


@st.cache_resource
def get_live_cam_component():
    """Write the small camera component to a temp folder and register it."""

    component_dir = Path(tempfile.gettempdir()) / "face_live_cam_component"
    component_dir.mkdir(parents=True, exist_ok=True)
    (component_dir / "index.html").write_text(LIVE_CAM_HTML, encoding="utf-8")

    return components.declare_component("face_live_cam", path=str(component_dir))


def decode_data_url(data_url):
    """Convert a 'data:image/jpeg;base64,...' string to a BGR OpenCV image."""

    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        return None

    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def _live_worker(holder, models):
    """Background thread: always predicts the newest frame (anti-spoofing + recognition).

    Keeps the heavy work out of Streamlit's script run, so slow predictions
    never block or starve the page. Stops itself when frames stop arriving.
    """

    last_id = None
    idle_since = time.time()

    while True:

        with holder["lock"]:
            frame_id = holder["frame_id"]
            data = holder["frame_data"]

        if frame_id is None or frame_id == last_id:
            if time.time() - idle_since > LIVE_IDLE_TIMEOUT:
                return
            time.sleep(0.05)
            continue

        idle_since = time.time()
        last_id = frame_id

        image_bgr = decode_data_url(data)
        if image_bgr is None:
            continue

        try:
            results = predict_image(image_bgr, models, use_anti_spoofing=True)
            error = None
        except Exception as exc:
            results = None
            error = str(exc)

        with holder["lock"]:
            holder["results"] = results
            holder["error"] = error


def run_live_camera(models):
    """Live Camera: anti-spoofing + recognition. The video stays clean,
    predictions update below it."""

    live_cam = get_live_cam_component()

    if "live_holder" not in st.session_state:
        st.session_state.live_holder = {
            "lock": threading.Lock(),
            "frame_id": None,
            "frame_data": None,
            "results": None,
            "error": None,
            "thread": None,
        }
    holder = st.session_state.live_holder

    # The component shows the camera and returns the newest frame every LIVE_INTERVAL_MS.
    frame = live_cam(
        interval_ms=LIVE_INTERVAL_MS,
        max_width=LIVE_MAX_WIDTH,
        key="live-cam",
        default=None
    )

    if not frame:
        with holder["lock"]:
            holder["frame_id"] = None
            holder["frame_data"] = None
            holder["results"] = None
            holder["error"] = None
        st.info("Click Start camera and allow camera access. Predictions appear here.")
        return

    with holder["lock"]:
        holder["frame_id"] = frame["id"]
        holder["frame_data"] = frame["data"]

    thread = holder["thread"]
    if thread is None or not thread.is_alive():
        thread = threading.Thread(
            target=_live_worker,
            args=(holder, models),
            daemon=True
        )
        holder["thread"] = thread
        thread.start()

    with holder["lock"]:
        results = holder["results"]
        error = holder["error"]

    if error:
        st.error(f"Prediction failed: {error}")
    elif results is None:
        st.info("Analyzing...")
    else:
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
