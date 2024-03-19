import cv2
import numpy as np
import math

def draw_lines_from_centroid(image, centroid, contour, angle_increment=5):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert centroid to integer coordinates
    centroid = (int(centroid[0]), int(centroid[1]))
    
    # Initialize list to store intersection points
    intersection_points = []
    
    # Iterate over angles from 0 to 360 degrees with specified increment
    for angle_degrees in range(0, 360, angle_increment):
        # Convert angle to radians
        angle_radians = math.radians(angle_degrees)
        
        # Define a line extending from centroid in the given angle direction
        line_end_x = int(centroid[0] + width * np.cos(angle_radians))
        line_end_y = int(centroid[1] + height * np.sin(angle_radians))
        
        # Draw the line on a blank image
        line_image = np.zeros_like(image)
        cv2.line(line_image, centroid, (line_end_x, line_end_y), (255, 255, 255), 1)
        
        # Find intersection points of the line with the contour
        _, _, intersections = cv2.findContours(cv2.bitwise_and(contour, line_image[:,:,0]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store intersection points
        if intersections:
            intersection_points.append(tuple(intersections[0][0][0]))
    
    # Draw lines from centroid to intersection points
    for intersection_point in intersection_points:
        cv2.line(image, centroid, intersection_point, (0, 255, 0), 1)

    return image

# Example usage
# Read the image
original_image = cv2.imread("data/nauplii/10_48_45_98660789.jpg", cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, binary_image = cv2.threshold(original_image, 230, 255, cv2.THRESH_BINARY)

inverted_image = cv2.bitwise_not(binary_image)

# Apply morphological closing to fill in small gaps
kernel = np.ones((15, 15), np.uint8)
closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

# Calculate centroid of the contour
M = cv2.moments(max_contour)
centroid_x = int(M["m10"] / M["m00"])
centroid_y = int(M["m01"] / M["m00"])
centroid = (centroid_x, centroid_y)

# Draw lines from centroid to contour
result_image = draw_lines_from_centroid(original_image.copy(), centroid, max_contour)

cv2.imshow("Lines from Centroid", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
