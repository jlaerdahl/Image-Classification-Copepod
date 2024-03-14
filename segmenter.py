import cv2
import numpy as np
import os

def segmenter(folder_path):
    idx = 0
    for path in os.listdir(folder_path):
        idx += 1
        image_path = folder_path+"/"+path

        if ".jpg" in image_path:
            original_image = cv2.imread(image_path)

            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Convert the image to grayscale
            _, binary_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)

            inverted_image = cv2.bitwise_not(binary_image)

            kernel = np.ones((15, 15), np.uint8)
            closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate over detected contours
            for i, contour in enumerate(contours):
                # Calculate the area of the contour
                area = cv2.contourArea(contour)
                
                # Filter out contours with small areas (noise)
                if area > 2000:
                    # Draw a bounding box around the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Crop the region inside the rectangle
                    cropped_region = original_image[y+2:y+h-2, x+2:x+w-2]
                    name = path.replace(".jpg", "")
                    # Save the cropped region as a new image
                    cv2.imwrite(f"needs_labeling/{name}{i}.jpg", cropped_region)

            # cv2.imwrite("needs_labeling/Segmented"+str(idx)+".jpg", original_image)

segmenter_test_data = "raw_images/11_04_35_215156"

segmenter(segmenter_test_data)
