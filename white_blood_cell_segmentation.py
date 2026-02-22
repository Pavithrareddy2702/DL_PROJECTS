import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from scipy.spatial.distance import directed_hausdorff

image_folder = "GroundTruths_bmpformat/Test_bmp"
gt_folder = "GroundTruths_bmpformat/Test_GT_bmp"

files = os.listdir(image_folder)
results = []

def crop_region(image, nucleus_gt, cytoplasm_gt):
    coords = np.column_stack(np.where(cytoplasm_gt > 0))

    if len(coords) == 0:
        return image, nucleus_gt, cytoplasm_gt

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (
        image[y_min:y_max, x_min:x_max],
        nucleus_gt[y_min:y_max, x_min:x_max],
        cytoplasm_gt[y_min:y_max, x_min:x_max]
    )


def hausdorff_distance(gt_mask, pred_mask):
    gt_points = np.column_stack(np.where(gt_mask > 0))
    pred_points = np.column_stack(np.where(pred_mask > 0))

    if len(gt_points) == 0 or len(pred_points) == 0:
        return 0

    return max(
        directed_hausdorff(gt_points, pred_points)[0],
        directed_hausdorff(pred_points, gt_points)[0]
    )

for file in files:
    print("Processing:", file)

    image = cv2.imread(os.path.join(image_folder, file))
    gt_image = cv2.imread(os.path.join(gt_folder, file), 0)

    if image is None or gt_image is None:
        continue

    nucleus_gt = (gt_image == 100).astype(np.uint8)
    cytoplasm_gt = (gt_image == 255).astype(np.uint8)

    image, nucleus_gt, cytoplasm_gt = crop_region(image, nucleus_gt, cytoplasm_gt)

    pixels = image.reshape((-1, 3)).astype(np.float32)

    #  KMeans 
    kmeans = KMeans(n_clusters=4, n_init=5)
    labels = kmeans.fit_predict(pixels)
    segmented = labels.reshape(image.shape[:2])

    k_nucleus = np.zeros_like(nucleus_gt)
    k_cytoplasm = np.zeros_like(cytoplasm_gt)

    best_n_overlap = 0
    best_c_overlap = 0

    for cluster in np.unique(segmented):
        mask = (segmented == cluster).astype(np.uint8)

        n_overlap = np.sum(mask * nucleus_gt)
        c_overlap = np.sum(mask * cytoplasm_gt)

        if n_overlap > best_n_overlap:
            best_n_overlap = n_overlap
            k_nucleus = mask

        if c_overlap > best_c_overlap:
            best_c_overlap = c_overlap
            k_cytoplasm = mask

    #  FCM 
    data = pixels.T
    centers, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, c=4, m=2, error=0.01, maxiter=150
    )

    f_labels = np.argmax(membership, axis=0)
    f_segmented = f_labels.reshape(image.shape[:2])

    f_nucleus = np.zeros_like(nucleus_gt)
    f_cytoplasm = np.zeros_like(cytoplasm_gt)

    best_n_overlap_f = 0
    best_c_overlap_f = 0

    for cluster in np.unique(f_segmented):
        mask = (f_segmented == cluster).astype(np.uint8)

        n_overlap = np.sum(mask * nucleus_gt)
        c_overlap = np.sum(mask * cytoplasm_gt)

        if n_overlap > best_n_overlap_f:
            best_n_overlap_f = n_overlap
            f_nucleus = mask

        if c_overlap > best_c_overlap_f:
            best_c_overlap_f = c_overlap
            f_cytoplasm = mask

    results.append({
        "Image": file,
        "KMeans_Nucleus_HD": hausdorff_distance(nucleus_gt, k_nucleus),
        "KMeans_Cytoplasm_HD": hausdorff_distance(cytoplasm_gt, k_cytoplasm),
        "FCM_Nucleus_HD": hausdorff_distance(nucleus_gt, f_nucleus),
        "FCM_Cytoplasm_HD": hausdorff_distance(cytoplasm_gt, f_cytoplasm),
    })


print("\nSegmentation Completed!\n")

df = pd.DataFrame(results)

print("Detailed Results:\n")
print(df)

print("\nAverage Performance:\n")
print(df.mean(numeric_only=True))
