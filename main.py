# realtime_yolo_opencv.py
import time
import cv2
from ultralytics import YOLO
import numpy as np

# -------- CONFIG --------
MODEL_PATH = "best.pt"   # path to your trained YOLO .pt
VIDEO_SOURCE = 0      # 0 for default webcam, or "path/to/video.mp4" or "rtsp://..."
CONF_THRESHOLD = 0.1
SHOW_FPS = True
WIN_NAME = "YOLOv8 - Fire & Smoke Detection (press q to quit)"
# ------------------------

# Load model
model = YOLO(MODEL_PATH)  # loads model to CPU or GPU automatically if available

# Optional: warm-up with a dummy image to avoid first-frame lag
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
try:
    _ = model(_dummy)  # warms up model
except Exception:
    pass

def draw_predictions(frame, boxes_xyxy, confs, classes, names):
    """
    Draw boxes on the frame (xyxy in pixel coords).
    boxes_xyxy: Nx4 numpy array [[x1,y1,x2,y2], ...]
    confs: N confidences
    classes: N class indices
    names: dict or list mapping class idx -> name
    """
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names[int(cls)] if isinstance(names, (list, dict)) else str(int(cls))
        text = f"{label} {conf:.2f}"

        # box
        color = (0, 0, 255) if label.lower().startswith("fire") else (255, 165, 0)  # fire red, smoke orange
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def main(video_source=VIDEO_SOURCE):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    # Get class names from model (if available)
    try:
        class_names = model.names  # dict: {0: 'fire', 1: 'smoke'}
    except Exception:
        class_names = None

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or couldn't read frame.")
            break

        # Convert BGR (OpenCV) to RGB for model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference (you can tune imgsz and augment options)
        # Using the model directly on np.ndarray returns a Results object
        results = model(img_rgb, imgsz=640, device=model.device, conf=CONF_THRESHOLD)  # device auto-chosen

        # results is a list (one per image). We processed one frame so get first result
        r = results[0]

        # Extract boxes, scores, class ids
        boxes = []
        scores = []
        class_ids = []
        if r.boxes is not None and len(r.boxes) > 0:
            # xyxy numpy: shape (N,4)
            boxes_np = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy.numpy()
            conf_np = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else r.boxes.conf.numpy()
            cls_np  = r.boxes.cls.cpu().numpy()  if hasattr(r.boxes.cls,  "cpu") else r.boxes.cls.numpy()

            boxes = boxes_np
            scores = conf_np
            class_ids = cls_np

        # Draw boxes
        draw_predictions(frame, boxes, scores, class_ids, class_names)

        # FPS calculation
        if SHOW_FPS:
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time)) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show frame
        cv2.imshow(WIN_NAME, frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
