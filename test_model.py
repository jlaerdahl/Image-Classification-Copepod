import cv2
import numpy as np
import random as rng

def threshold_image(image_path, threshold_value):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)

    inverted_image = cv2.bitwise_not(binary_image)

    # Apply morphological closing to fill in small gaps
    kernel = np.ones((15, 15), np.uint8)
    closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

    # # Display the original and binary images
    # cv2.imshow("Original Image", original_image)
    # cv2.imshow("Binary Image", inverted_image)
    # cv2.imshow("Closed Image", closed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # minEllipse = [None]*len(contours)
    # for i, c in enumerate(contours):
    #     if c.shape[0] > 10:
    #         minEllipse[i] = cv2.fitEllipse(c)

    drawing = np.zeros((closed_image.shape[0], closed_image.shape[1], 3), dtype=np.uint8)

    max_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

    ellipse = cv2.fitEllipse(max_contour)

    return ellipse, max_contour, closed_image

    # cv2.ellipse(drawing, ellipse, (0, 0, 255), 2)

    # object_inside_ellipse, _ = checkPercentage(closed_image, ellipse)

    # best_percentage = 0

    # # Modify the ellipse with the rotation angle
    # while best_percentage < 90:

    #     best_ellipse_rot, best_percentage = rot_ellipse(closed_image, ellipse, max_contour, best_percentage)
    #     best_ellipse_pos, best_percentage = repos_ellipse(closed_image, best_ellipse_rot, max_contour, best_percentage)
    #     ellipse, best_percentage = resize_ellipse(closed_image, best_ellipse_pos, max_contour, best_percentage)

    #     print(best_percentage)

    #     cv2.ellipse(drawing, ellipse, (0, 0, 255), 2)

    #     cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

    #     # Display the rotated ellipse
    #     cv2.imshow("Best so far", drawing)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # # object_inside_ellipse, _ = checkPercentage(closed_image, ellipse)

    # cv2.ellipse(drawing, ellipse, (0, 0, 255), 2)

    # cv2.imshow("Closed Image", drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resize_ellipse(image, best_ellipse, max_contour, best_percentage):
    _, check_ratio_inside_pre = checkPercentage(image, best_ellipse)

    print("Pre: ", check_ratio_inside_pre)

    resize_ellipse = (best_ellipse[0], (best_ellipse[1][0], best_ellipse[1][1]-5), best_ellipse[2])
    object_inside_ellipse, check_ratio_inside = checkPercentage(image, resize_ellipse)

    print("Post: ", check_ratio_inside)

    if object_inside_ellipse > best_percentage and check_ratio_inside > check_ratio_inside_pre:
        best_percentage = object_inside_ellipse
        best_ellipse = resize_ellipse
        print("1")
        return best_ellipse, best_percentage
    
    resize_ellipse = (best_ellipse[0], (best_ellipse[1][0], best_ellipse[1][1]+5), best_ellipse[2])
    object_inside_ellipse, check_ratio_inside = checkPercentage(image, resize_ellipse)

    if object_inside_ellipse > best_percentage and check_ratio_inside > check_ratio_inside_pre:
        best_percentage = object_inside_ellipse
        best_ellipse = resize_ellipse
        print("2")
        return best_ellipse, best_percentage

    resize_ellipse = (best_ellipse[0], (best_ellipse[1][0]-5, best_ellipse[1][1]), best_ellipse[2])
    object_inside_ellipse, check_ratio_inside = checkPercentage(image, resize_ellipse)

    if object_inside_ellipse > best_percentage and check_ratio_inside > check_ratio_inside_pre:
        best_percentage = object_inside_ellipse
        best_ellipse = resize_ellipse
        print("3")
        return best_ellipse, best_percentage
    
    resize_ellipse = (best_ellipse[0], (best_ellipse[1][0]+5, best_ellipse[1][1]), best_ellipse[2])
    object_inside_ellipse, check_ratio_inside = checkPercentage(image, resize_ellipse)

    if object_inside_ellipse > best_percentage and check_ratio_inside > check_ratio_inside_pre:
        best_percentage = object_inside_ellipse
        best_ellipse = resize_ellipse
        print("4")
        return best_ellipse, best_percentage

    drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    cv2.ellipse(drawing, resize_ellipse, (0, 0, 255), 2)

    cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

    object_inside_ellipse, _ = checkPercentage(image, resize_ellipse)

    # Keep track of the best ellipse based on percentage
    if object_inside_ellipse > best_percentage:
        print(best_percentage)
        best_percentage = object_inside_ellipse
        best_ellipse = resize_ellipse

    return best_ellipse, best_percentage


def repos_ellipse(image, best_ellipse, max_contour, best_percentage):
    best_ellipse = best_ellipse

    # Modify the ellipse with the rotation angle
    for posx in range(0, image.shape[0], 5):
        for posy in range(0, image.shape[1], 5):
            repos_ellipse = ((posx, posy), best_ellipse[1], best_ellipse[2])

            drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

            cv2.ellipse(drawing, repos_ellipse, (0, 0, 255), 2)

            cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

            object_inside_ellipse, _ = checkPercentage(image, repos_ellipse)
        
            # Keep track of the best ellipse based on percentage
            if object_inside_ellipse > best_percentage:
                best_percentage = object_inside_ellipse
                best_ellipse = repos_ellipse

    return best_ellipse, best_percentage

def rot_ellipse(image, ellipse, max_contour, best_percentage):
    best_ellipse = ellipse

    # Rotate the ellipse
    for rotation in range(0, 180, 5):
        # Modify the ellipse with the rotation angle
        rotated_ellipse = (ellipse[0], ellipse[1], ellipse[2] + rotation)

        drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        cv2.ellipse(drawing, rotated_ellipse, (0, 0, 255), 2)

        cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

        object_inside_ellipse, _ = checkPercentage(image, rotated_ellipse)

        # # Display the rotated ellipse
        # cv2.imshow("Rotated ", drawing)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Keep track of the best ellipse based on percentage
        if object_inside_ellipse > best_percentage:
            best_percentage = object_inside_ellipse
            best_ellipse = rotated_ellipse

    return best_ellipse, best_percentage

def checkPercentage(image, ellipse):
    # Create a mask for the region inside the ellipse
    mask_ellipse = np.zeros_like(image)
    cv2.ellipse(mask_ellipse, ellipse, 255, thickness=cv2.FILLED)

    # Apply the mask to isolate the region inside the ellipse
    object_inside_ellipse = cv2.bitwise_and(mask_ellipse, image)

    # Calculate the percentage of the ellipse filled with parts of the object
    total_pixels_of_object_inside_ellipse = np.sum(object_inside_ellipse > 0)

    # Calculate the total area of the ellipse
    total_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)

    # Calculate the filled area ratio
    filled_area_ratio = (total_pixels_of_object_inside_ellipse / total_ellipse_area) * 100

    # print(f"Percentage of the ellipse filled with parts of the object: {filled_area_ratio:.2f}%")

    return filled_area_ratio, total_pixels_of_object_inside_ellipse


def length_calc(ellipse, max_contour, image):
    # Extract major and minor axes
    major_axis, minor_axis = ellipse[1]

    # Calculate orientation
    orientation = ellipse[2]

    # Sort contour points based on their angle
    contour_points_sorted = sorted(max_contour, key=lambda point: np.arctan2(point[0][1] - ellipse[0][1], point[0][0] - ellipse[0][0]))

    # Place ellipse on contour
    new_major_axis_endpoint = contour_points_sorted[-1][0]
    new_minor_axis_endpoint = contour_points_sorted[-2][0]

    # Create a new ellipse with the sorted points
    new_ellipse = ((new_major_axis_endpoint[0], new_minor_axis_endpoint[1]), (major_axis, minor_axis), orientation)
    
    drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Calculate ellipse points
    ellipse_points = cv2.ellipse2Poly(new_ellipse[0], (int(new_ellipse[1][0] / 2), int(new_ellipse[1][1] / 2)), int(new_ellipse[2]), 0, 360, 10)

    # Draw the rotated ellipse
    cv2.polylines(drawing, [ellipse_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw the contour
    cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Replace 'your_image_path.jpg' with the path to your image
# Replace 'your_threshold_value' with the desired threshold value
ellipse, max_contour, closed_image = threshold_image("data/nauplii/11_04_27_210448_3.jpg", 230)

length_calc(ellipse, max_contour, closed_image)
