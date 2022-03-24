import cv2
import dlib
import numpy as np
from PIL import Image

SHAPE_PREDICTOR_FILE_PATH = 'shape_predictor_68_face_landmarks.dat'
FACIAL_POINTS = 68


def detect_facial_landmarks(source_img, target_img):
    print("Doing facial landmarks detection")

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILE_PATH)

    points_sum_array = np.zeros((FACIAL_POINTS, 2))

    points_list_source = []
    points_list_target = []

    curr_image_list = points_list_source
    for image in (source_img, target_img):

        # image size
        width = image.shape[0]
        length = image.shape[1]

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        rectangles = detector(image)

        # loop over the face detections
        for (k, rect) in enumerate(rectangles):

            # determine the facial landmarks for the face region
            shape = predictor(image, rect)

            for i in range(0, FACIAL_POINTS):
                x = shape.part(i).x
                y = shape.part(i).y

                curr_image_list.append((x, y))
                points_sum_array[i][0] += x
                points_sum_array[i][1] += y

            # corners + centers to each image's list
            add_corners_and_centers(curr_image_list, width, length)

            # set current image to target
            curr_image_list = points_list_target

    # corresp -> sum of every point from img1 and img2

    # Add back the background
    # corners and centers to the resulting array
    average_points_list = points_sum_array / 2  # center between two image points
    average_points_list = np.append(average_points_list, [[1, 1]], axis=0)
    average_points_list = np.append(average_points_list, [[length - 1, 1]], axis=0)
    average_points_list = np.append(average_points_list, [[(length - 1) // 2, 1]], axis=0)
    average_points_list = np.append(average_points_list, [[1, width - 1]], axis=0)
    average_points_list = np.append(average_points_list, [[1, (width - 1) // 2]], axis=0)
    average_points_list = np.append(average_points_list, [[(length - 1) // 2, width - 1]], axis=0)
    average_points_list = np.append(average_points_list, [[length - 1, width - 1]], axis=0)
    average_points_list = np.append(average_points_list, [[(length - 1), (width - 1) // 2]], axis=0)

    return [(width, length), points_list_source, points_list_target, average_points_list]


def add_corners_and_centers(image_points, width, length):
    # corners
    image_points.append((1, 1))
    image_points.append((length - 1, 1))
    image_points.append((1, width - 1))
    image_points.append((length - 1, width - 1))

    # centers
    image_points.append(((length - 1) // 2, 1))
    image_points.append((1, (width - 1) // 2))
    image_points.append(((length - 1) // 2, width - 1))
    image_points.append(((length - 1), (width - 1) // 2))


def draw_facial_landmarks(image, points, output):
    for (x, y) in points:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    res = Image.fromarray(cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB))
    res.save("facial-landmarks/" + output)
