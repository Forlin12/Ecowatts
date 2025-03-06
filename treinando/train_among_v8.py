from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../resources/yolo11n.pt")
    results = model.train(data="among.yaml", epochs=30, imgsz=640)
