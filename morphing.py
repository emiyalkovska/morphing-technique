from subprocess import Popen, PIPE

import cv2
import numpy as np
from PIL import Image


def morph_images(source_image, target_image, duration, frame_rate, size):
    # todo add dynamic output name
    output = "cross-dissolve-images/output.mp4"
    sequence_length = int(duration * frame_rate)

    p = Popen(
        ['ffmpeg', '-y', '-f', 'image2pipe',
         '-r', str(frame_rate),
         '-s', str(size[1]) + 'x' + str(size[0]), '-i', '-',
         '-c:v', 'libx264', '-crf', '25', '-vf',
         'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p',
         output], stdin=PIPE)

    for img_index in range(0, sequence_length):
        # Convert Mat to float data type
        source_image = np.float32(source_image)
        target_image = np.float32(target_image)

        alpha = img_index / (sequence_length - 1)

        # Alpha blend rectangular patches
        morphed_frame = (1.0 - alpha) * source_image + alpha * target_image

        # convert back to image
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(p.stdin, 'JPEG')
        res.save("cross-dissolve-images/img_" + str(img_index) + ".JPEG")
