from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from color import get_color_and_shade, get_coloradd_symbol
from util import put_unicode_text

# your imports: ultralytics YOLO, color.py functions, etc.
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
# ... copy your MIN_COLOR_CONFIDENCE, get_color_and_shade, etc.

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read image
    img_bytes = file.read()
    img = Image.open(BytesIO(img_bytes))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Optional: resize for speed
    frame = cv2.resize(frame, (640, 640))

    # Run your detection (copy-paste from your mobile_mode.py logic)
    results = model(frame, conf=0.52, iou=0.45, verbose=False)[0]
    bounding_boxes = []

    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cls_id = int(det.cls[0])
        name = model.names[cls_id]
        conf = float(det.conf)

        # your color detection code here...
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        color_name, color_conf = get_color_and_shade(roi)
        if color_conf < 0.22:
            continue

        symbol = get_coloradd_symbol(color_name)
        label = f"{name} {int(conf * 100)}% {color_name} {symbol}"

        bounding_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})

    return jsonify({"bounding_boxes": bounding_boxes})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=8000, debug=True
    )  # debug=True is fine for school/testing
