import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rng

def measure_object_length_area(inverted_image, pixel_size, full_image_width_pixels):
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

    # print(f"Length of object: {length_cm} micrometers")
    # print(f"Area of object: {areaContour} micrometers")

    # Draw the contour on the original image for visualization
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)

    drawing = np.zeros((inverted_image.shape[0], inverted_image.shape[1], 3), dtype=np.uint8)
    
    # ellipse = [None]*len(max_contour)

    ellipse = cv2.fitEllipse(max_contour)

    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

    cv2.ellipse(drawing, ellipse, color, 2)

    # Display the result
    cv2.imshow("Object Measurement", image)
    cv2.imshow("Ellipse", drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ellipse, image

# Example usage:
folder_path = "data/nauplii"

image_file = "nauplii.jpg"

# Read the image
image = cv2.imread(image_file)

height, width = image.shape[:2]

pixel_size = 0.75  # Replace with the known pixel size of the image in a chosen unit (e.g., centimeters per pixel)
full_image_width_pixels = width  # Replace with the full width of the image in pixels

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask (adjust threshold as needed)
_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

# Assuming 'binary_image' is your binary image (black and white)
inverted_image = cv2.bitwise_not(thresh)

obj, max_contour = measure_object_length_area(inverted_image, pixel_size, full_image_width_pixels)
