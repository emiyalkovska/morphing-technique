from subprocess import Popen, PIPE

import cv2
import numpy as np
from PIL import Image


# Apply affine transform calculated using src_tri and inter_tri
def affine_transform(img, src_tri, tar_tri, size):
    # find affine transform
    trans_matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(tar_tri))

    width = size[0]
    height = size[1]

    # apply transformation
    output_triangle = cv2.warpAffine(img, trans_matrix, (width, height), None,
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
    # return transformed triangle
    return output_triangle


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_tri1 = affine_transform(img1Rect, t1Rect, tRect, size)
    warp_tri2 = affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warp_tri1 + alpha * warp_tri2

    # Copy triangular region of the rectangular patch to the output image
    morphed_frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = morphed_frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
            1 - mask) + imgRect * mask


def generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, triangles_indexes, size, output):
    num_images = int(duration * frame_rate)
    p = Popen(
        ['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', str(size[1]) + 'x' + str(size[0]), '-i', '-',
         '-c:v', 'libx264', '-crf', '25', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', output],
        stdin=PIPE)

    # for every intermediate image
    for j in range(0, num_images):

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        weighted_points = []
        alpha = j / (num_images - 1)

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            weighted_points.append((x, y))

        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)

        for i in range(len(triangles_indexes)):
            x = int(triangles_indexes[i][0])
            y = int(triangles_indexes[i][1])
            z = int(triangles_indexes[i][2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            # final point between source and target
            t = [weighted_points[x], weighted_points[y], weighted_points[z]]

            # Morph one triangle at a time.
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)

            pt1 = (int(t[0][0]), int(t[0][1]))
            pt2 = (int(t[1][0]), int(t[1][1]))
            pt3 = (int(t[2][0]), int(t[2][1]))

            cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)

        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(p.stdin, 'JPEG')
        res.save("morphed_images/img_" + str(j) + ".JPEG")

    p.stdin.close()
    p.wait()

