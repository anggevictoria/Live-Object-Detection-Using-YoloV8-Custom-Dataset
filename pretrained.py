import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Ensure the correct model filename

# Open the default webcam
cap = cv2.VideoCapture(0)

# Loop through the video frames from the webcam
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.predict(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Webcam Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the frame could not be read
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
