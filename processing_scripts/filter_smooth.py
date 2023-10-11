import cv2
import numpy as np
import os

def compute_avg_histogram(img_dir):
    sum_hist = [np.zeros(256), np.zeros(256), np.zeros(256)]
    total_images = 0

    for img_file in os.listdir(img_dir):
        if img_file.startswith('0') and img_file.endswith('.jpg'):
            total_images += 1
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)

            # Apply Gaussian blur for smoothing
            img = cv2.GaussianBlur(img, (3, 3), 0)

            for channel in range(3):
                hist = cv2.calcHist([img], [channel], None, [256], [0,256])
                sum_hist[channel] += hist[:,0]

    avg_hist = [h/total_images for h in sum_hist]
    return avg_hist

def hist_match(source, template_histograms):
    matched = source.copy()
    for channel in range(3):
        matched[:,:,channel] = cv2.equalizeHist(source[:,:,channel])
        source_hist = cv2.calcHist([matched], [channel], None, [256], [0,256])
        source_cdf = source_hist.cumsum()
        template_cdf = template_histograms[channel].cumsum()
        lut = np.interp(source_cdf, template_cdf, np.arange(256))
        matched[:,:,channel] = cv2.LUT(matched[:,:,channel], lut.astype(np.uint8))
    
    # Apply Gaussian blur for smoothing
    matched = cv2.GaussianBlur(matched, (3, 3), 0)
    return matched

if __name__ == '__main__':
    source_dir = "no_person.class/resized_no_people"
    target_dir = "filtered/no_people"
    
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Compute the average histogram from images starting with 0
    template_histograms = compute_avg_histogram(source_dir)

    # Apply the color tone to images starting with 9 and save to the new directory
    for img_file in os.listdir(source_dir):
        if img_file.startswith('9') and img_file.endswith('.jpg'):
            source_path = os.path.join(source_dir, img_file)
            source = cv2.imread(source_path)
            transfer = hist_match(source, template_histograms)
            target_path = os.path.join(target_dir, img_file)
            cv2.imwrite(target_path, transfer)
