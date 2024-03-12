import cv2
import numpy as np

# Load your image
image = cv2.imread("data/nauplii/11_04_27_210448_3.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding or any other method to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the max contour (you might already have this step)
max_contour = max(contours, key=cv2.contourArea)

# Approximate the contour
epsilon = 0.02 * cv2.arcLength(max_contour, True)
approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)

drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

# # Get the bounding rectangle
# x, y, w, h = cv2.boundingRect(approx_contour)

# # Crop the image
# crop_img = image[y:y+h, x:x+w]

# # Fit ellipse to the cropped image
# ellipse = cv2.fitEllipse(crop_img)

# # Draw the ellipse on the original image
# cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# Display the result
cv2.imshow('Ellipse Fitted to Copepod Body', drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()
