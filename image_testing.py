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

def findBackground(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve the results of background subtraction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Apply background subtraction
    foreground_mask = bg_subtractor.apply(blur)

    # Invert the mask to get the background
    background_mask = cv2.bitwise_not(foreground_mask)

    # Create an all-white image
    background = np.ones_like(image, np.uint8) * 255

    # Put the original image where the mask is white
    background[background_mask == 255] = image[background_mask == 255]

    return background

def findRatio(ellipse, img, max_contour):
    # Extract ellipse parameters
    center = ellipse[0]
    axes = ellipse[1]
    rotation = ellipse[2]

    rot_overlap_percentage = []
    cen_overlap_percentage = []
    ax_overlap_percentage = []

    rotations = list(range(0,180 ,10))
    for i, rotated in enumerate(rotations):
        ellipse = (center, axes, rotated)
        # Create a black mask of the same size as the object image
        mask = np.zeros_like(img)

        # Draw the ellipse on the mask
        cv2.ellipse(mask, ellipse, color=255, thickness=-1)

        intersection = cv2.bitwise_and(img, mask)
        rot_overlap_percentage.append(np.sum(intersection) / np.sum(mask) * 100)

        # Display the result (for visualization purposes)
        cv2.imshow('Ellipse Mask', mask)
        cv2.imshow('Object Image', img)
        cv2.imshow('Intersection', intersection)
        # cv2.imshow('Contour', max_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    rot = rotations[rot_overlap_percentage.index(max(rot_overlap_percentage))]

    ax = list(range(30,300,30))
    for i, axis in enumerate(ax):
        test_ax = (axis, axis*0.5)
        ellipse = (center, test_ax, rotation)
        # Create a black mask of the same size as the object image
        mask = np.zeros_like(img)

        # Draw the ellipse on the mask
        cv2.ellipse(mask, ellipse, color=255, thickness=-1)

        intersection = cv2.bitwise_and(img, mask)
        ax_overlap_percentage.append(np.sum(intersection) / np.sum(mask) * 100)

        # Display the result (for visualization purposes)
        cv2.imshow('Ellipse Mask', mask)
        cv2.imshow('Object Image', img)
        cv2.imshow('Intersection', intersection)
        # cv2.imshow('Contour', max_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(ax_overlap_percentage)

    axises = ax[ax_overlap_percentage.index(max(ax_overlap_percentage))]

    print(axises)

    # centers = list(range(0,min(img.shape),10))
    # for i, centered in enumerate(centers):
    #     centered = (centered, centered)
    #     ellipse = (centered, axes, rotation)
    #     # Create a black mask of the same size as the object image
    #     mask = np.zeros_like(img)

    #     # Draw the ellipse on the mask
    #     cv2.ellipse(mask, ellipse, color=255, thickness=-1)

    #     intersection = cv2.bitwise_and(img, mask)
    #     cen_overlap_percentage.append(np.sum(intersection) / np.sum(mask) * 100)

    #     # Display the result (for visualization purposes)
    #     cv2.imshow('Ellipse Mask', mask)
    #     cv2.imshow('Object Image', img)
    #     cv2.imshow('Intersection', intersection)
    #     # cv2.imshow('Contour', max_contour)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # cen = centers[cen_overlap_percentage.index(max(cen_overlap_percentage))]

    overlap_percentage = []
    ellipse = (center, (axises, axises), rot)
    # Create a black mask of the same size as the object image
    mask = np.zeros_like(img)

    # Draw the ellipse on the mask
    cv2.ellipse(mask, ellipse, color=255, thickness=-1)

    intersection = cv2.bitwise_and(img, mask)
    overlap_percentage.append(np.sum(intersection) / np.sum(mask) * 100)

    # Display the result (for visualization purposes)
    cv2.imshow('Ellipse Mask', mask)
    cv2.imshow('Object Image', img)
    cv2.imshow('Intersection', intersection)
    # cv2.imshow('Contour', max_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"The ratio of object pixels inside the ellipse: {max(overlap_percentage)}")


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

# findRatio(obj, inverted_image, max_contour)
