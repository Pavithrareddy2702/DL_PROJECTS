import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_niblack, threshold_sauvola


BASE_PATH = "training"

IMAGE_DIR = os.path.join(BASE_PATH, "images")
GT_DIR = os.path.join(BASE_PATH, "1st_manual")
MASK_DIR = os.path.join(BASE_PATH, "mask")


def compute_sensitivity(gt, pred, mask):

    TP = 0
    FN = 0

    rows = gt.shape[0]
    cols = gt.shape[1]

    for i in range(rows):
        for j in range(cols):

            if mask[i][j] == 1:
                if gt[i][j] == 1:
                    if pred[i][j] == 1:
                        TP = TP + 1
                    else:
                        FN = FN + 1

    if (TP + FN) == 0:      # to avoid division by zero
        return 1.0

    sensitivity = TP / (TP + FN)

    return sensitivity


niblack_scores = []
sauvola_scores = []

image_files = os.listdir(IMAGE_DIR)
image_files = sorted(image_files)

for index, file in enumerate(image_files):


    image_path = os.path.join(IMAGE_DIR, file)
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green_channel = image[:, :, 1]   # to extract green channel bcz it shows best contrast between vessels and background


    gt_name = file.replace("_training.tif", "_manual1.gif")
    gt_path = os.path.join(GT_DIR, gt_name)

    gt_image = io.imread(gt_path)
    gt_image = np.squeeze(gt_image)

    if len(gt_image.shape) == 3:
        gt_image = gt_image[:, :, 0]

    gt_binary = np.zeros_like(gt_image)

    for i in range(gt_image.shape[0]):
        for j in range(gt_image.shape[1]):

            if gt_image[i][j] > 0:
                gt_binary[i][j] = 1
            else:
                gt_binary[i][j] = 0


 
    mask_name = file.replace("_training.tif", "_training_mask.gif")
    mask_path = os.path.join(MASK_DIR, mask_name)

    mask_image = io.imread(mask_path)
    mask_image = np.squeeze(mask_image)

    if len(mask_image.shape) == 3:
        mask_image = mask_image[:, :, 0]

    mask_binary = np.zeros_like(mask_image)

    for i in range(mask_image.shape[0]):
        for j in range(mask_image.shape[1]):
            if mask_image[i][j] > 0:
                mask_binary[i][j] = 1
            else:
                mask_binary[i][j] = 0


    niblack_threshold = threshold_niblack(
        green_channel,
        window_size=25,
        k=0.2
    )

    niblack_binary = np.zeros_like(green_channel)

    for i in range(green_channel.shape[0]):
        for j in range(green_channel.shape[1]):
            if green_channel[i][j] < niblack_threshold[i][j]:
                niblack_binary[i][j] = 1
            else:
                niblack_binary[i][j] = 0


    sauvola_threshold = threshold_sauvola(
        green_channel,
        window_size=25,
        k=0.2
    )

    sauvola_binary = np.zeros_like(green_channel)

    for i in range(green_channel.shape[0]):
        for j in range(green_channel.shape[1]):
            if green_channel[i][j] < sauvola_threshold[i][j]:
                sauvola_binary[i][j] = 1
            else:
                sauvola_binary[i][j] = 0


    sens_niblack = compute_sensitivity(gt_binary,
                                       niblack_binary,
                                       mask_binary)

    sens_sauvola = compute_sensitivity(gt_binary,
                                       sauvola_binary,
                                       mask_binary)

    niblack_scores.append(sens_niblack)
    sauvola_scores.append(sens_sauvola)
    

print("Average Sensitivity (Niblack):", np.mean(niblack_scores))
print("Average Sensitivity (Sauvola):", np.mean(sauvola_scores))