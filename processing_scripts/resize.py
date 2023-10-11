import os
from PIL import Image

def resize_images_in_folder(folder_path, output_size=(96, 96)):
    # Create a 'resized_people' subfolder if it doesn't exist
    output_folder = os.path.join(folder_path, 'resized_no_people')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            img_path = os.path.join(folder_path, file_name)

            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if the image has an alpha channel
                    if img.mode == 'RGBA':
                        img = img.convert("RGB")
                    img_resized = img.resize(output_size)
                    
                    output_path = os.path.join(output_folder, file_name)
                    img_resized.save(output_path)
                    
                    print(f"Resized: {file_name}")
            except Exception as e:
                print(e)

# Specify the folder containing the .jpg images
folder_path = 'no_person.class'
resize_images_in_folder(folder_path)