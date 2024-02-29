import cv2
import os
import numpy as np

def measure_object_length_area(image_path, pixel_size, full_image_width_pixels):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask (adjust threshold as needed)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Assuming 'binary_image' is your binary image (black and white)
    inverted_image = cv2.bitwise_not(thresh)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assuming it's the object)
    max_contour = max(contours, key=cv2.contourArea)

    # Calculate the length of the object in pixels
    length_pixels = cv2.arcLength(max_contour, True)

    # Calculate the area of the object
    areaContour = cv2.contourArea(max_contour)

    # Calculate the length of the object in a chosen unit (e.g., centimeters)
    length_cm = (length_pixels / full_image_width_pixels) * pixel_size

    print(f"Length of object: {length_cm} micrometers")
    print(f"Area of object: {areaContour} micrometers")

    # Draw the contour on the original image for visualization
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Object Measurement", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
folder_path = "data/nauplii"

image_file = "11_06_44_690979_1.jpg"

image_path = os.path.join(folder_path, image_file)

image = cv2.imread(image_path)

height, width = image.shape[:2]

pixel_size = 0.75  # Replace with the known pixel size of the image in a chosen unit (e.g., centimeters per pixel)
full_image_width_pixels = width  # Replace with the full width of the image in pixels

measure_object_length_area(image_path, pixel_size, full_image_width_pixels)
