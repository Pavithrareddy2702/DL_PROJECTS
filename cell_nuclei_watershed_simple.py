import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = "stage1_train 2"


def combine_masks(mask_path, height, width):


    final_mask = np.zeros((height, width), dtype=np.uint8)

    mask_list = os.listdir(mask_path)

    for mask_name in mask_list:

        full_mask_path = os.path.join(mask_path, mask_name)
        single_mask = cv2.imread(full_mask_path, 0)
        final_mask = np.maximum(final_mask, single_mask)

  
    for i in range(height):
        for j in range(width):
            if final_mask[i][j] > 0:
                final_mask[i][j] = 255
            else:
                final_mask[i][j] = 0

    return final_mask


def watershed_simple(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    ret_value, binary = cv2.threshold(
        gray,
        0,
        255,
        threshold_type
    )

    kernel = np.ones((3, 3), np.uint8)   # to remove small nois

    clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    dist = cv2.distanceTransform(
        clean,
        cv2.DIST_L2,
        5
    )

    max_value = dist.max()

    thresh_value = 0.7 * max_value


    ret2, sure_fg = cv2.threshold(
        dist,
        thresh_value,
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)

 
    markers = markers + 1

    markers = cv2.watershed(image, markers)
    h, w = gray.shape
    output = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if markers[i][j] > 1:
                output[i][j] = 255
            else:
                output[i][j] = 0

    unique_regions = np.unique(markers)
    count = len(unique_regions) - 2   #to remove boundary and background

    return output, count


def watershed_with_markers(image, gt_mask):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(
        gt_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    dist = cv2.distanceTransform(
        clean_mask,
        cv2.DIST_L2,
        5
    )

    max_value = dist.max()

    thresh_value = 0.5 * max_value

    ret3, sure_fg = cv2.threshold(
        dist,
        thresh_value,
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers = cv2.watershed(image, markers)

    h, w = gray.shape
    output = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if markers[i][j] > 1:
                output[i][j] = 255
            else:
                output[i][j] = 0

    unique_regions = np.unique(markers)
    count = len(unique_regions) - 2

    return output, count

folder_list = os.listdir(BASE_PATH)

for index in range(5):

    folder_name = folder_list[index]

    full_path = os.path.join(BASE_PATH, folder_name)

    image_folder = os.path.join(full_path, "images")
    mask_folder = os.path.join(full_path, "masks")

    image_name = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    height = image.shape[0]
    width = image.shape[1]

    gt_mask = combine_masks(mask_folder, height, width)

    ground_truth_count = len(os.listdir(mask_folder))

    result_without, count_without = watershed_simple(image)

    result_with, count_with = watershed_with_markers(image, gt_mask)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(2, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(2, 3, 4)
    plt.imshow(result_without, cmap="gray")
    plt.title("Without Markers")

    plt.subplot(2, 3, 5)
    plt.imshow(result_with, cmap="gray")
    plt.title("With Markers")

    plt.tight_layout()
    plt.show()

    print("Ground Truth Count:", ground_truth_count)
    print("Without Markers Count:", count_without)
    print("With Markers Count:", count_with)
    print("-------------------------------------------")