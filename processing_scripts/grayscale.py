import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

def get_dominant_colors(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    colors = kmeans.cluster_centers_
    return colors

def apply_colors(image, colors):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=len(colors)).fit(pixels)
    new_pixels = np.array([colors[label] for label in kmeans.labels_])
    new_image = new_pixels.reshape(image.shape)
    return new_image

def process_images():
    source_directory = "no_person.class/resized_no_people"
    target_directory = "recolored/no_people"
    dominant_colors = None
    
    # Check if target directory exists, if not create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Extract colors from images starting with 0
    for filename in os.listdir(source_directory):
        if filename.startswith("0") and filename.endswith(".jpg"):
            image_path = os.path.join(source_directory, filename)
            # Copy the image to the target directory
            shutil.copy(image_path, target_directory)
            
            image = cv2.imread(image_path)
            dominant_colors = get_dominant_colors(image)
            break

    # If no dominant colors are found, exit
    if dominant_colors is None:
        print("No reference image found for color extraction.")
        return

    # Apply colors to images starting with 9
    for filename in os.listdir(source_directory):
        if filename.startswith("9") and filename.endswith(".jpg"):
            image_path = os.path.join(source_directory, filename)
            image = cv2.imread(image_path)
            new_image = apply_colors(image, dominant_colors)
            new_image_path = os.path.join(target_directory, filename)
            cv2.imwrite(new_image_path, new_image)

if __name__ == "__main__":
    process_images()
