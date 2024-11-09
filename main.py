from detectObject import YOLOObjectDetection

def main():
    # Parse arguments and start the detection system
    detector = YOLOObjectDetection()
    args = detector.parse_arguments()
    detection_system = YOLOObjectDetection(resolution=args.webcam_resolution)
    detection_system.start_detection()

if __name__ == "__main__":
    main()
