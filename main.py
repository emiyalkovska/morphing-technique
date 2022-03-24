import argparse

import cv2

from facial_landmarks import detect_facial_landmarks
from morphing import cross_dissolve
from facial_landmarks import draw_facial_landmarks


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourceImg", required=True, help="source image path should be present")
    parser.add_argument("--targetImg", required=True, help="target image path should be present")
    parser.add_argument("--duration", type=int, default=1, help="duration of the output sequence (in seconds)")
    parser.add_argument("--frameRate", type=int, default=5, help="frame-rate of the output sequence")
    parser.add_argument("--technique", type=str, default="alpha-blend")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # load source and target image from path
    source_img = cv2.imread(args.sourceImg)
    target_img = cv2.imread(args.targetImg)

    img_size = (source_img.shape[0], source_img.shape[1])

    # [size, points_source, points_target, average_points_list] = detect_facial_landmarks(source_img, target_img)

    # draw facial points
    # draw_facial_landmarks(source_img, points_source, "source_points.JPEG")
    # draw_facial_landmarks(target_img, points_target, "target_points.JPEG")

    # cross_dissolve(source_img, target_img, args.duration, args.frameRate, img_size)
