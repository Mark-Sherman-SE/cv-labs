import numpy as np
import cv2
from numba import njit
import time


@njit
def orig_jit_median_filter(image, ksize=5):
    edge = ksize // 2
    width, height = image.shape[:2]
    image_blur = np.zeros((width, height), dtype=np.uint8)
    for x in range(edge, width - edge):
        for y in range(edge, height - edge):
            window = np.zeros(ksize * ksize, dtype=np.uint8)
            i = 0
            for w_x in range(ksize):
                for w_y in range(ksize):
                    window[i] = image[x + w_x - edge, y + w_y - edge]
                    i += 1
            window = np.sort(window)

            image_blur[x, y] = window[ksize * ksize // 2]
    return image_blur


def orig_median_filter(image, ksize=5):
    edge = ksize // 2
    width, height = image.shape[:2]
    image_blur = np.zeros((width, height), dtype=np.uint8)
    for x in range(edge, width - edge):
        for y in range(edge, height - edge):
            window = np.zeros(ksize * ksize, dtype=np.uint8)
            i = 0
            for w_x in range(ksize):
                for w_y in range(ksize):
                    window[i] = image[x + w_x - edge, y + w_y - edge]
                    i += 1
            window = np.sort(window)

            image_blur[x, y] = window[ksize * ksize // 2]
    return image_blur.astype(np.uint8)

# center = ksize ** 2 // 2
#             if ksize % 2 == 0:
#                 res = (window[center] + window[center + 1]) // 2
#             else:
#                 res = window[center]

# temp = np.zeros((ksize, ksize))
# for w_x in range(ksize):
#     for w_y in range(ksize):
#         temp[w_x, w_y] = image[x + w_x - edge, y + w_y - edge]
# temp = np.sort(temp)
# image_blur[x, y] = temp[edge / 2, edge / 2]


def process(path, func, ksize=5, show=True):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    show_orig = True
    frame_sum = 0
    start_video_time = time.time()
    while success:
        frame_sum += 1
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_frame = func(frame, ksize)
        if show:
            if cv2.waitKey(10) == ord('w'):
                show_orig = not show_orig
            frame_to_show = frame if show_orig else processed_frame
            cv2.imshow('frame', frame_to_show)
        success, image = vidcap.read()
    end_video_time = time.time()
    vidcap.release()
    cv2.destroyAllWindows()
    print("Video finished")
    process_time = end_video_time - start_video_time
    return process_time, process_time / frame_sum


if __name__ == "__main__":
    video_process_time, frame_process_time = process("test_video_2_sec_small.mp4", cv2.medianBlur, ksize=7, show=False)
    print("Opencv frame process time:", frame_process_time)
    print("Opencv video process time:", video_process_time)

    video_process_time, frame_process_time = process("test_video_2_sec_small.mp4", orig_jit_median_filter, ksize=7,
                                                     show=False)
    print("Orig JIT frame process time:", frame_process_time)
    print("Orig JIT video process time:", video_process_time)

    video_process_time, frame_process_time = process("test_video_2_sec_small.mp4", orig_median_filter, ksize=7,
                                                     show=False)
    print("Orig frame process time:", frame_process_time)
    print("Orig video process time:", video_process_time)
