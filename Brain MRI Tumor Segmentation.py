import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_sauvola


img_folder = "dataset/images"
mask_folder = "dataset/masks"


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    total_pixels = np.sum(y_true) + np.sum(y_pred)

    if total_pixels == 0:   # to avoid zero-division error
        return 1.0

    return (2.0 * intersection) / total_pixels

def jaccard_score(y_true, y_pred):
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter

    if union == 0:
        return 1.0

    return inter / union


def otsu_threshold(image):
    threshold_value = threshold_otsu(image)
    binary_result = np.where(image > threshold_value, 1, 0)
    return binary_result


def sauvola_threshold(img):

    window_size = 25
    k_value = 0.02

    threshold_matrix = threshold_sauvola(
        img,
        window_size=window_size,
        k=k_value,
        r=128
    )

    binary_output = np.where(img > threshold_matrix, 1, 0)

    return binary_output



def show_results(original, mask, otsu_result, sauvola_result):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(1, 4, 3)
    plt.imshow(otsu_result, cmap='gray')
    plt.title("Otsu")

    plt.subplot(1, 4, 4)
    plt.imshow(sauvola_result, cmap='gray')
    plt.title("Sauvola")

    plt.tight_layout()
    plt.show()



def main():

    image_files = os.listdir(img_folder)

    dice_otsu_scores = []
    dice_sauvola_scores = []

    jaccard_otsu_scores = []
    jaccard_sauvola_scores = []

    for idx, file in enumerate(image_files):

        print("Processing:", file)

        image_path = os.path.join(img_folder, file)
        mask_path = os.path.join(mask_folder, file)

        image = cv2.imread(image_path, 0) #Reading being done as grayscale
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            continue

      
        mask = np.where(mask > 0, 1, 0)  #modify mask from grayscale to binary


        otsu_result = otsu_threshold(image)

        sauvola_result = sauvola_threshold(image)


        dice_otsu_value = dice_score(mask, otsu_result)
        dice_sauvola_value = dice_score(mask, sauvola_result)

        dice_otsu_scores.append(dice_otsu_value)
        dice_sauvola_scores.append(dice_sauvola_value)

        jaccard_otsu_value = jaccard_score(mask, otsu_result)
        jaccard_sauvola_value = jaccard_score(mask, sauvola_result)

        jaccard_otsu_scores.append(jaccard_otsu_value)
        jaccard_sauvola_scores.append(jaccard_sauvola_value)



        if idx == 1:
            show_results(image, mask, otsu_result, sauvola_result)

  
    print("FINAL AVERAGE RESULTS")
    print("Average Dice (Otsu):", np.mean(dice_otsu_scores))
    print("Average Dice (Sauvola):", np.mean(dice_sauvola_scores))
    print("Average Jaccard (Otsu):", np.mean(jaccard_otsu_scores))
    print("Average Jaccard (Sauvola):", np.mean(jaccard_sauvola_scores))


if __name__ == "__main__":
    main()
