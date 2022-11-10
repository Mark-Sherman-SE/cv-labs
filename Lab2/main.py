import os
import numpy as np
import cv2
import pandas as pd


def match_template(path_to_image, path_to_template, threshold=0.8):
    img_rgb = cv2.imread(path_to_image)
    img_rgb_orig = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(path_to_template, 0)

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    # Draw a rectangle around the matched region.
    if loc[0].size > 0:
        m = [0, 0]
        for pt in zip(*loc[::-1]):
            m = [max(m[0], pt[0]), max(m[1], pt[1])]
        cv2.rectangle(img_rgb, m, (m[0] + w, m[1] + h), (255, 0, 0), 15)
    else:
        img_rgb = None

    return img_rgb_orig, img_rgb


def apply_sift(path_to_image, path_to_template, min_match_count=10):
    img_rgb = cv2.imread(path_to_image)
    img_rgb_orig = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(path_to_template, 0)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    des1 = np.uint8(des1 * 255)
    des2 = np.uint8(des2 * 255)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    point_list = []
    for m in matches[:min_match_count]:
        point_list.append(m.queryIdx)
    c_list = cv2.KeyPoint_convert(kp1)
    needed_c_list = []
    for p in point_list:
        needed_c_list.append(c_list[p])

    df = pd.DataFrame(needed_c_list, columns=['x', 'y'])
    y_max, y_min = int(df.y.max()), int(df.y.min())
    x_max, x_min = int(df.x.max()), int(df.x.min())

    img_rgb = cv2.rectangle(img_rgb, (x_max, y_max), (x_min, y_min), color=(255, 55, 0), thickness=15)
    return img_rgb_orig, img_rgb


if __name__ == "__main__":
    templates = ["1_crop", "8_crop", "10_crop", "2", "3", "4", "5", "6", "7", "9"]
    main_images = ["1", "8", "10"]

    params = [["matchTemplate", dict(threshold=0.5)], ["SIFT", dict(min_match_count=10)]]

    algorithms = [match_template, apply_sift]
    for alg, (name, param) in zip(algorithms, params):
        for image in main_images:
            image_path = os.path.join("data/orig", f"{image}.jpg")
            for template in templates:
                template_path = os.path.join("data/orig", f"{template}.jpg")
                _, res = alg(image_path, template_path, **param)
                if res is not None:
                    cv2.imwrite(os.path.join("data", name, f"img_{image}_tmpl_{template}.jpg"), res)
