from ultralytics import YOLO

def train_model():
    model = YOLO('/Users/grigorcrandon/Desktop/yolo11n.pt')

    model.train(
        data='data.yaml',
        epochs=200, # try 150
        imgsz=640,
        workers=4,
        batch=16,#batch try 32
        device = 'mps',
        name='merged_paintings_sculptures'
    )

if __name__ == "__main__":
    train_model()

