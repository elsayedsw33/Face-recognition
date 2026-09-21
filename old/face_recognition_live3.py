import cv2
import numpy as np
import onnxruntime as ort

# =========================
# Load database
# =========================

face_database = np.load(
    "face_database.npy",
    allow_pickle=True
).item()

threshold = float(
    np.load("best_threshold.npy")
)


# =========================
# Load models
# =========================

detector = cv2.FaceDetectorYN_create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.60,
    nms_threshold=0.30,
    top_k=5000
)

rec = cv2.FaceRecognizerSF_create(
    "face_recognition_sface_2021dec.onnx",
    ""
)

# Load the quantized anti-spoofing ONNX model using ONNX Runtime
spoof_sess = ort.InferenceSession("best_model.onnx")
spoof_input_name = spoof_sess.get_inputs()[0].name


# =========================
# Recognition
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
# Camera
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1280, 720)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4].astype(int)
            
            # ---------------------------------------------------------
            # 1. Expand Bounding Box (20% Margin)
            # ---------------------------------------------------------
            margin_w = int(fw * 0.2)
            margin_h = int(fh * 0.2)
            
            # Ensure crop boundaries with margin do not exceed frame dimensions
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(w, x + fw + margin_w)
            y2 = min(h, y + fh + margin_h)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            # ---------------------------------------------------------
            # 2. Anti-Spoofing Preprocessing (PyTorch Standard)
            # ---------------------------------------------------------
            # Resize and convert to float32
            resized_crop = cv2.resize(face_crop, (128, 128))
            img_float = resized_crop.astype(np.float32) / 255.0
            
            # Convert BGR (OpenCV default) to RGB (PyTorch default)
            img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
            
            # Apply ImageNet Mean and Standard Deviation
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_rgb - mean) / std
            
            # Reshape from (Height, Width, Channels) to (Channels, Height, Width)
            blob = np.transpose(img_normalized, (2, 0, 1))
            
            # Add Batch dimension: (1, 3, 224, 224)
            blob = np.expand_dims(blob, axis=0)
            
            # ---------------------------------------------------------
            # 3. Inference and Classification
            # ---------------------------------------------------------
            # Run inference using ONNX Runtime
            spoof_preds = spoof_sess.run(None, {spoof_input_name: blob})[0]
            
            scores = spoof_preds[0] if len(spoof_preds.shape) > 1 else spoof_preds
            predicted_class = np.argmax(scores)
            
            #print(f"Scores: {scores} | Predicted Class: {predicted_class}")
            
            # IMPORTANT: Change to `== 0` if the model marks real faces as 0 instead of 1
            is_real = (predicted_class == 1) 
            
            # ---------------------------------------------------------
            # 4. Face Recognition (Only if Real)
            # ---------------------------------------------------------
            if is_real:
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
                    continue

                embedding = embedding / norm

                # Recognize
                name, score = recognize_embedding(
                    embedding,
                    face_database,
                    threshold
                )
                
                label = f"{name} | {score:.2f}"
                box_color = (0, 255, 0) # Green for real faces
            
            else:
                label = "Spoof Detected"
                box_color = (0, 0, 255) # Red for spoofed faces

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                box_color,
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()