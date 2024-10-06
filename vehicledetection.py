import cv2
import numpy as np

def get_coordinates(vehicle, frame_shape):
    height, width = frame_shape[:2]
    x_center, y_center, box_width, box_height = vehicle[0:4] * np.array([width, height, width, height])
    x = int(x_center - box_width / 2)
    y = int(y_center - box_height / 2)
    w = int(box_width)
    h = int(box_height)
    return x, y, w, h

def is_crossing_line(centroid_y, line_position=300):
    # Define the position of the counting line (e.g., at y=300)
    return centroid_y > line_position

# Initialize video capture
cap = cv2.VideoCapture('videos/traffic.mp4')
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
layer_names = net.getLayerNames()

# Use the correct indexing for the layer names
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

vehicle_count = 0
crossed_vehicle_ids = set()  # To store the IDs of vehicles that have crossed the line

# Line position for counting
line_position = 300

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to make it smaller (e.g., 50% of original size)
    frame_resized = cv2.resize(frame, (800, 600))  # Adjust the width and height as needed

    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for i, vehicle in enumerate(detection):
            confidence = vehicle[5]
            if confidence > 0.5:
                x, y, w, h = get_coordinates(vehicle, frame_resized.shape)
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the centroid of the vehicle
                centroid_x = x + w // 2
                centroid_y = y + h // 2

                # Draw the counting line
                cv2.line(frame_resized, (0, line_position), (frame_resized.shape[1], line_position), (0, 0, 255), 2)

                # If the vehicle is crossing the line and hasn't been counted yet
                if is_crossing_line(centroid_y, line_position) and i not in crossed_vehicle_ids:
                    vehicle_count += 1
                    crossed_vehicle_ids.add(i)  # Add the vehicle's ID to the set

    # Display the vehicle count on the frame
    cv2.putText(frame_resized, f"Vehicle Count: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display resized frame with detections
    cv2.imshow("Vehicle Detection", frame_resized)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Total Vehicles Detected: ", vehicle_count)
