# YOLO Criminal Detection

Criminal detection system using YOLOv8 with Flask web interface.

## Features

- YOLOv8 object detection
- Web interface and REST API
- Docker support
- Visual detection with bounding boxes

## Prerequisites

- Python 3.10+
- Docker (optional)
- GPU with CUDA (optional, for training)

## Installation

### Local Setup

```bash
git clone <repository-url>
cd yolo-criminal-detection
pip install -r requirements.txt
python app.py
```

Access at `http://localhost:8080`

### Docker

```bash
docker build -t yolo-criminal-detection .
docker run -p 8080:8080 yolo-criminal-detection
```

## Training

Use the Jupyter notebook `fine_tune_YOLO.ipynb` to train the model.

### Dataset Structure

```
criminal_yolo/
├── data.yaml
├── train/images/ + train/labels/
├── valid/images/ + valid/labels/
└── test/images/ + test/labels/
```

**data.yaml:**
```yaml
path: ./content/criminal_yolo
train: train/images
val: valid/images
nc: 2
names: ['person', 'criminal']
```

Labels format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

### Training Steps

1. Install: `!pip install ultralytics`
2. Load model: `model = YOLO("yolov8l.pt")`
3. Train:
```python
model.train(
    data='./content/criminal_yolo/data.yaml',
    name='criminal_yolo',
    epochs=100,
    imgsz=640,
    batch=8,
    freeze=15,
    dropout=0.3,
    device=0
)
```
4. Export: `cp runs/detect/criminal_yolo/weights/best.pt model.pt`

**Tips:** Reduce batch size if out of memory. Use GPU for training (Google Colab recommended).

## Project Structure

```
yolo-criminal-detection/
├── app.py                    # Flask application
├── model.pt                  # Trained YOLO model
├── fine_tune_YOLO.ipynb     # Training notebook
├── requirements.txt
├── Dockerfile
├── templates/index.html      # Web interface
└── content/criminal_yolo/   # Training data (not in repo)
```

## API

- **GET** `/` - Web interface
- **POST** `/predict` - Upload image, returns annotated image

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8080/predict --output result.jpg
```

## Configuration

Edit `app.py` for model settings:
```python
results = model(img, conf=0.5, iou=0.45, imgsz=640)
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)

