import os
import cv2
import numpy as np

def enhance_brightness_contrast(image, alpha=1.2, beta=30):
    """Enhance brightness and contrast of the image."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def sharpen_image(image):
    """Apply sharpening to the image."""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def process_people_images():
    '''source_directory = "no_person.class/resized_no_people"
    target_directory = "edited_images/contrast/no_people"'''

    source_directory = "person.class/resized_people"
    target_directory = "edited_images/contrast/people"
    
    # Check if target directory exists, if not create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(source_directory, filename)
            image = cv2.imread(image_path)

            # Enhance and sharpen the image
            enhanced_image = enhance_brightness_contrast(image)
            final_image = sharpen_image(enhanced_image)

            # Save the processed image
            output_path = os.path.join(target_directory, filename)
            cv2.imwrite(output_path, final_image)

if __name__ == "__main__":
    process_people_images()
