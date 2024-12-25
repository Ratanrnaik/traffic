import cv2
import numpy as np

# Load the reference image (the image to compare against)
reference_image_path = r"C:\Users\Lenovo\Desktop\show\ALLVEHICLES.png"
reference_image = cv2.imread(reference_image_path)

# Load the image containing cars (the image to be analyzed)
cars_image_path = r"C:\Users\Lenovo\Desktop\show\6.png"
cars_image = cv2.imread(cars_image_path)

# Check if the images were loaded successfully
if reference_image is None or cars_image is None:
    print("Error: Unable to load one or both of the images.")
    exit()

# Convert images to grayscale for template matching
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
cars_image_gray = cv2.cvtColor(cars_image, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(cars_image_gray, reference_image_gray, cv2.TM_CCOEFF_NORMED)

# Get the maximum similarity score and its location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Convert the similarity score to a percentage (multiply by 100)
accuracy_percentage = max_val * 100

# Load the YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load the image and get its width and height
image = cv2.imread(cars_image_path)
height, width = image.shape[:2]

# Define the confidence threshold for detection
confidence_threshold = 0.3  # Adjust this threshold as needed

# Create a blob from the image and set it as the input to the network
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get the output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through the network to get detections
detections = net.forward(layer_names)

# Initialize a counter for detected vehicles
vehicle_count = 0

# Create a copy of the image to draw bounding boxes
output_image = image.copy()

# Loop over the detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            # Check if the detected object belongs to a vehicle class
            if classes[class_id] in ["car", "truck", "bus", "bicycle"]:
                # Calculate the coordinates of the bounding box
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                
                # Draw the bounding box and label on the output image
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Increment the vehicle count
                vehicle_count += 1

# Draw the accuracy percentage and vehicle count on the output image
text = f"Accuracy: {accuracy_percentage:.2f}%"
cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Resize the output image to 500x500 pixels
output_image = cv2.resize(output_image, (500, 500))

# Print the number of detected vehicles
print(f"Number of Vehicles Detected: {vehicle_count}")

# Display the output image with object detection, accuracy percentage, and vehicle count
cv2.imshow("Image with Object Detection and Accuracy", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
S