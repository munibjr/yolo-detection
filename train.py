from ultralytics import YOLO
import argparse

def train_yolo(data_yaml, epochs=100):
    model = YOLO('yolov11n.pt')  # nano model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        save=True,
        name='damage_detector'
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    train_yolo(args.data, args.epochs)
