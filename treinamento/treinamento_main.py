from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../resources/yolo11n.pt")
    results = model.train(data="among.yaml", epochs=1, imgsz=640)
