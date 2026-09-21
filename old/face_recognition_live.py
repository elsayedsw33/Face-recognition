
import cv2
import numpy as np


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

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

            # Draw label
            label = f"{name} | {score:.2f}"

            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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
