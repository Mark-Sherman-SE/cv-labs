import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def match_template(path_to_image, path_to_template, threshold=0.8):
    img_rgb = cv2.imread(path_to_image)
    img_rgb_orig = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template_rgb = cv2.imread(path_to_template)
    template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    # Draw a rectangle around the matched region.
    if loc[0].size > 0:
        m = [0, 0]
        for pt in zip(*loc[::-1]):
            m = [max(m[0], pt[0]), max(m[1], pt[1])]
        cv2.rectangle(img_rgb, m, (m[0] + w, m[1] + h), (255, 0, 0), 15)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.axis('off')
        ax1.set_title('Original image')
        ax1.imshow(img_rgb)

        ax2.axis('off')
        ax2.set_title('Template image')
        ax2.imshow(template_rgb)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()

        #.reshape(canvas.get_width_height()[::-1] + (3,))
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
    else:
        img_rgb = None
        image = None

    return img_rgb_orig, img_rgb, image


def apply_sift(path_to_image, path_to_template, min_match_count=10):
    img_rgb = cv2.imread(path_to_image)
    img_rgb_orig = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template_rgb = cv2.imread(path_to_template)
    template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

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
    #imgage_keypoints = cv.drawKeypoints(img_rgb, kp1, None, color=(0,255,0), flags=0)
    # images_matches = cv2.drawMatches(img_rgb, kp1, template_rgb, kp2, matches[:min_match_count], None,
    #                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # images_matches = cv2.cvtColor(images_matches, cv2.COLOR_BGR2RGB)
    #
    # plt.figure(figsize=(8, 6))
    # plt.title(path_to_image + " " + path_to_template)
    # plt.imshow(images_matches)
    # plt.show()

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
    images_matches = cv2.drawMatches(img_rgb, kp1, template_rgb, kp2, matches[:min_match_count], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_rgb_orig, img_rgb, images_matches


if __name__ == "__main__":
    templates = ["1_crop", "8_crop", "10_crop", "2", "3", "4", "5", "6", "7", "9"]
    main_images = ["1", "8", "10"]

    params = [["matchTemplate", dict(threshold=0.5)], ["SIFT", dict(min_match_count=10)]]

    algorithms = [match_template, apply_sift]
    for alg, (name, param) in zip(algorithms, params):
        print(1)
        for image in main_images:
            image_path = os.path.join("data/orig", f"{image}.jpg")
            for template in templates:
                template_path = os.path.join("data/orig", f"{template}.jpg")
                _, res, res_match = alg(image_path, template_path, **param)
                if res_match is not None:
                    cv2.imwrite(os.path.join("data", name, f"img_{image}_tmpl_{template}.jpg"), res_match)
