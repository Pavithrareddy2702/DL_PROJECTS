import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "stage1_train 2"


def combine_masks(folder, h, w):
    mask = np.zeros((h, w), np.uint8)
    for f in os.listdir(folder):
        m = cv2.imread(os.path.join(folder, f), 0)
        mask = np.maximum(mask, m)
    mask[mask > 0] = 255
    return mask


def watershed(image, mask=None):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)

    if mask is None:
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        base = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 2)
    else:
        base = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)

    dist = cv2.distanceTransform(base, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(base, kernel, 3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    output = np.zeros(gray.shape, np.uint8)
    output[markers > 1] = 255

    count = len(np.unique(markers)) - 2

    return output, count


folders = os.listdir(BASE_PATH)

for i in range(5):

    folder_path = os.path.join(BASE_PATH, folders[i])

    img_folder = os.path.join(folder_path, "images")
    mask_folder = os.path.join(folder_path, "masks")

    img_name = os.listdir(img_folder)[0]
    image = cv2.imread(os.path.join(img_folder, img_name))

    h, w = image.shape[:2]

    gt_mask = combine_masks(mask_folder, h, w)
    gt_count = len(os.listdir(mask_folder))

    result_without, count_without = watershed(image)
    result_with, count_with = watershed(image, gt_mask)

    plt.figure(figsize=(10,5))

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(2,2,2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(2,2,3)
    plt.imshow(result_without, cmap="gray")
    plt.title(f"Without ({count_without})")

    plt.subplot(2,2,4)
    plt.imshow(result_with, cmap="gray")
    plt.title(f"With ({count_with})")

    plt.tight_layout()
    plt.show()

    print("GT:", gt_count,
          "| Without:", count_without,
          "| With:", count_with)
    print("--------------------------------")