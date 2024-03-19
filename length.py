import cv2
import numpy as np
import random as rng
import math

def threshold_image(image_path, threshold_value):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)

    inverted_image = cv2.bitwise_not(binary_image)

    # Apply morphological closing to fill in small gaps
    kernel = np.ones((15, 15), np.uint8)
    closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((closed_image.shape[0], closed_image.shape[1], 3), dtype=np.uint8)

    max_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

    ellipse = cv2.fitEllipse(max_contour)

    cv2.ellipse(drawing, ellipse, (0, 0, 255), 2)

    # cv2.imshow("Closed Image", drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return closed_image, max_contour, ellipse


def find_centroid(closed_image, max_contour):
    drawing = np.zeros((closed_image.shape[0], closed_image.shape[1], 3), dtype=np.uint8)

    # Calculate moments of the contour
    M = cv2.moments(max_contour)

    # Calculate centroid coordinates
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    centroid = (centroid_x, centroid_y)

    # print("Centroid coordinates (x, y):", centroid)

    return centroid


def draw_lines_from_centroid(image, centroid, contour, angle_increment):
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
            
        line_end = (line_end_x, line_end_y)
        
        # Draw the line on a blank image
        # line_image = np.zeros_like(image, dtype=np.uint8) 
        # line = cv2.line(line_image, centroid, line_end, (255, 255, 255), 1)
        # cv2.circle(line_image, centroid, 5, (255, 255, 255), -1)  # Green color marker
        # cv2.drawContours(line_image, [contour], -1, (255, 255, 255), 2)
        # cv2.imshow("Centroid", line_image)
        # cv2.waitKey(0)                               
        # cv2.destroyAllWindows()

        line_points = get_line_points(centroid, line_end)

        contour_points = []

        for i in range(len(contour)-1):
            point1 = (contour[i][0][0], contour[i][0][1])
            point2 = (contour[i+1][0][0], contour[i+1][0][1])
            contour_points.append(get_line_points(point1, point2))
        
        contour_points = [item for sublist in contour_points for item in sublist]

        points = []

        for pnt in line_points:
            if pnt in contour_points:
                points.append(pnt)
        
        distances = []
        if len(points) > 1:
            for pnt in points:
                distances.append(math.sqrt((pnt[0] - centroid[0])**2 + (pnt[1] - centroid[1])**2))
            intersection_points.append(points[distances.index(min(distances))])
            # print("intersection point: ", intersection_points[-1])
        elif len(points) == 1:
            intersection_points.append(points[0])
            # print("intersection point: ", intersection_points[-1])
        else:
            intersection_points.append(centroid)
            # print("no intersection point ")
    
    # for elem in intersection_points:
    #     new_line_image = np.zeros_like(image, dtype=np.uint8) 
    #     cv2.circle(new_line_image, elem, 5, (255, 255, 0), -1)  # Green color marker
    #     cv2.drawContours(new_line_image, [contour], -1, (255, 255, 255), 2)
    #     cv2.imshow("Centroid", new_line_image)
    #     cv2.waitKey(0)                               
    #     cv2.destroyAllWindows()

    return intersection_points

def find_length_of_line(intersection_points, angle_increment):
    lengths = []
    for i in range(int((360/angle_increment)/2)):
        point1 = intersection_points[i]
        point2 = intersection_points[i+int((360/angle_increment)/2)]
        lengths.append(math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2))
    index = lengths.index(max(lengths)) 

    return intersection_points[index], intersection_points[index+int((360/angle_increment)/2)]

# Bresenham's Line Algorithm
def get_line_points(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine increments for x and y
    if x1 < x2:
        sx = 1
    else:
        sx = -1
    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    line_points = []
    while True:
        line_points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x1 = x1 + sx
        if e2 < dx:
            err = err + dx
            y1 = y1 + sy

    return line_points


angle_increment = 3

closed_image, max_contour, ellipse = threshold_image("data/copepod/10_52_50_559099148.jpg", 230)

centroid = find_centroid(closed_image, max_contour)

intersection_points = draw_lines_from_centroid(closed_image, centroid, max_contour, angle_increment)

max_point = find_length_of_line(intersection_points, angle_increment)

print(max_point)
new_line_image = np.zeros_like(closed_image, dtype=np.uint8) 
cv2.drawContours(new_line_image, [max_contour], -1, (255, 255, 255), 2)
cv2.line(new_line_image, max_point[0], max_point[1], (255, 255, 255), 1)
cv2.imshow("Centroid", new_line_image)
cv2.waitKey(0)                               
cv2.destroyAllWindows()
