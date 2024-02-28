import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import shutil
import math

pixelsize = 0.75 #micrometers

folder_path = "data/nauplii"

image_file = "11_06_44_690979_1.jpg"

image_path = os.path.join(folder_path, image_file)

image = imread(image_path)

height, width = image.shape[:2]

# Print the dimensions
print(f"Image Dimensions: {width} x {height} micrometers")

fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title(image_file)
ax.axis('off')
plt.show()