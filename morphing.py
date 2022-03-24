from subprocess import Popen, PIPE

import cv2
import numpy as np
from PIL import Image


def triangulation(f_w, f_h, narray):
    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary.
    narray = narray.tolist()
    points = [(int(x[0]), int(x[1])) for x in narray]
    # print(points)
    dictionary = {x[0]: x[1] for x in list(zip(points, range(76)))}

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    triangle_list = subdiv.getTriangleList()

    tri_points_indexes = []
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        tri_points_indexes.append((dictionary[pt1], dictionary[pt2], dictionary[pt3]))

    return tri_points_indexes


def cross_dissolve(source_image, target_image, duration, frame_rate, size):
    output = "cross-dissolve-images/output.mp4"
    frames = int(duration * frame_rate)

    p = Popen(
        ['ffmpeg', '-y', '-f', 'image2pipe',
         '-r', str(frame_rate),
         '-s', str(size[1]) + 'x' + str(size[0]), '-i', '-',
         '-c:v', 'libx264', '-crf', '25', '-vf',
         'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p',
         output], stdin=PIPE)

    # Convert Mat to float data type
    source_image = np.float32(source_image)
    target_image = np.float32(target_image)

    for t in range(0, frames):
        opacity = t / (frames - 1)

        # Alpha blend two images
        frame_t = (1.0 - opacity) * source_image + opacity * target_image

        # convert back to image
        res = Image.fromarray(cv2.cvtColor(np.uint8(frame_t), cv2.COLOR_BGR2RGB))
        res.save(p.stdin, 'JPEG')
        res.save("cross-dissolve-images/img_" + str(t) + ".JPEG")
