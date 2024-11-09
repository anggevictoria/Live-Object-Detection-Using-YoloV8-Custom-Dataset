import cv2
import argparse
from ultralytics import YOLO

class YOLOObjectDetection:
    def __init__(self, model_path="Model/yolov8m.pt", resolution=(640, 480)):
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.frame_width, self.frame_height = resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument(
            "--webcam-resolution", 
            default=[640, 480], 
            nargs=2, 
            type=int
        )
        args = parser.parse_args()
        return args
    
    def plot_boxes(self, results, frame):
        xyxys = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
        for xyxy in xyxys:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        return frame
    
    def start_detection(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame, show=True)
            frame = self.plot_boxes(results, frame)
            # Uncomment if you want to manually display frames
            # cv2.imshow("YOLOv8 Live Detection", frame)
            if cv2.waitKey(30) == 27:  # Exit on 'Esc' key
                break
