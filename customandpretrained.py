import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
pretrained_model = YOLO("yolov8m.pt")  # Ensure the correct model filename

# Load the custom YOLO model
custom_model = YOLO("customModelm.pt")  # Ensure the correct model filename

# Open the default webcam
cap = cv2.VideoCapture(0)

# Loop through the video frames from the webcam
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run inference with both models
        results_pretrained = pretrained_model.predict(frame)
        results_custom = custom_model.predict(frame)

        # Plot results from both models on the same frame
        annotated_frame = frame.copy()
        annotated_frame = results_pretrained[0].plot(annotated_frame)
        annotated_frame = results_custom[0].plot(annotated_frame)

        # Display the combined annotated frame
        cv2.imshow("YOLO Combined Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the frame could not be read
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
