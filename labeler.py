import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import shutil

def on_key_press(event):
    global image, image_path
    if event.key == '1':
        image_path_to_move = image_path  # Replace with the actual path to your image
        destination_folder = 'data/copepod'  # Replace with the desired destination folder
        move_image_to_folder(image_path_to_move, destination_folder)
        plt.close()

    elif event.key == '2':
        image_path_to_move = image_path  # Replace with the actual path to your image
        destination_folder = 'data/nauplii'  # Replace with the desired destination folder
        move_image_to_folder(image_path_to_move, destination_folder)
        plt.close()
    
    elif event.key == '3':
        image_path_to_move = image_path  # Replace with the actual path to your image
        destination_folder = 'data/egg'  # Replace with the desired destination folder
        move_image_to_folder(image_path_to_move, destination_folder)
        plt.close()

    elif event.key == '4':
        image_path_to_move = image_path  # Replace with the actual path to your image
        destination_folder = 'data/unknown'  # Replace with the desired destination folder
        move_image_to_folder(image_path_to_move, destination_folder)
        plt.close()

 
def plot_images_in_folder(folder_path):
    global image, image_file, image_path
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter only image files (you may want to adjust this based on the types of images you have)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Iterate through image files and plot each one
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Read and plot the image
        image = imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(image_file)

        # Add text under the image
        text_to_add = "1: Copepod, 2: Nauplii, 3: Egg, 4: Unknown"
        ax.text(0.5, -0.1, text_to_add, transform=ax.transAxes,
                fontsize=12, color='white', ha='center', va='center', backgroundcolor='black')

        # Remove the axes for a cleaner look (optional)
        ax.axis('off')


        # Connect the keypress event handler
        plt.connect('key_press_event', on_key_press)

        # Display the image and wait for user input
        plt.show()

def move_image_to_folder(image_path, destination_folder):
    global image_file
    # Construct the destination path by joining the destination_folder and image_file_name
    destination_path = os.path.join(destination_folder, image_file)

    print(image_path)
    print(destination_path)

    try:
        # Move the image to the destination folder
        shutil.copy(image_path, destination_path)
        print(f"Image '{image_file}' moved to '{destination_folder}'.")
    except Exception as e:
        print(f"Error: Unable to move the image. {e}")

# Replace 'your_folder_path' with the path to the folder containing your images
folder_path = 'E:/C-Feed/27_02_2024_Data_gathering/27_02_2024_Data_gathering_1_nauplii_full_test_03'
plot_images_in_folder(folder_path)
