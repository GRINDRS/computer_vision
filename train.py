from ultralytics import YOLO

def train_model():
    model = YOLO('/Users/grigorcrandon/Desktop/yolo11n.pt')

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        workers=4,
        batch=16,
    )

if __name__ == "__main__":
    train_model()

