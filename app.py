from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO("model.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Veuillez envoyer une image avec le champ 'file'"}), 400

    file = request.files["file"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image invalide"}), 400

    results = model(img)[0]

    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label = f"{model.names[cls]}: {conf:.2f}"

        rectangle_color = (0, 255, 100)
        if label.split(":")[0].strip().lower() == "criminal":
            rectangle_color = (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, 2)

        font_scale = 0.4
        font_thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_y1 = max(y1 - label_height - 10, 0)
        label_y2 = y1

        cv2.rectangle(img, (x1, label_y1), (x1 + label_width + 10, label_y2), rectangle_color, -1)

        cv2.putText(img, label, (x1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    _, img_encoded = cv2.imencode('.jpg', img)

    return Response(img_encoded.tobytes(), mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
