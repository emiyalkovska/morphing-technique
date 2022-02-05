# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

import cv2

from morphing import morph_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourceImg", required=True, help="source image path should be present")
    parser.add_argument("--targetImg", required=True, help="target image path should be present")
    parser.add_argument("--duration", type=int, default=2, help="duration of the output sequence (in seconds)")
    parser.add_argument("--frameRate", type=int, default=5, help="frame-rate of the output sequence")
    parser.add_argument("--technique", type=str, default="alpha-blend")

    return parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = parse_arguments()

    # load source and target image from path
    source_img = cv2.imread(args.sourceImg)
    target_img = cv2.imread(args.targetImg)

    img_size = (source_img.shape[0], source_img.shape[1])

    morph_images(source_img, target_img, args.duration, args.frameRate, img_size)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
