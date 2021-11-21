import json
import os

import cv2
import dlib
import numpy as np
from imutils import face_utils


def create_video_from_images(image_folder, output):
    images = [img for img in os.listdir(image_folder) if img.startswith("0-")]

    def select_digit(e):
        import re
        pattern = "-(.*?).png"
        substring = re.search(pattern, e).group(1)
        return int(substring)

    images.sort(key=select_digit)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def get_facial_landmarks(shape_predictor_path, image_path, visualization=False):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rectangles = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rectangles):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if visualization:
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    if visualization:
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        return shape


def optical_flow(image, landmarks=None, visualization=False):
    cap = cv2.VideoCapture(image)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    if landmarks is not None:
        p0 = np.array(landmarks).reshape((-1, 1, 2))
    else:
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=50,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    if visualization:
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        while 1:
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, np.float32(p0), None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            k = cv2.waitKey(1000) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    else:
        # we save the first image
        saved_pixels = [p0.reshape(-1, 2)]
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is not False:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, np.float32(p0), None, **lk_params)

                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    # we save the pixels of the tracked features
                    saved_pixels.append(np.floor(good_new))
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                if not p0.shape[0] == good_new.shape[0]:
                    assert False, "A tracked point was lost"
                p0 = good_new.reshape(-1, 1, 2)
            else:
                return np.asarray(saved_pixels)


def color_to_depth_pixels(color_pixels, json_file):
    with open(json_file) as f:
        data = json.load(f)
    rgb_camera_properties = {"pp": data["streams"][0]["properties"]["intrinsics"]["pp"],
                             "focal": data["streams"][0]["properties"]["intrinsics"]["focal"][0],
                             "scale": data["streams"][0]["properties"]["intrinsics"]["scale"]}
    depth_camera_properties = {"pp": data["streams"][1]["properties"]["intrinsics"]["pp"],
                               "focal": data["streams"][1]["properties"]["intrinsics"]["focal"][0]}
    return np.floor((depth_camera_properties["focal"] / rgb_camera_properties["focal"]) * (
            color_pixels - rgb_camera_properties["pp"]) + depth_camera_properties["pp"]).astype(
        int), rgb_camera_properties


def depth_pixels_to_distances(depth_pixels, image_folder, rgb_camera_properties):
    distances = np.empty((depth_pixels.shape[0], depth_pixels.shape[1]))
    for i in range(depth_pixels.shape[0]):
        image = cv2.imread(image_folder + "/1-" + str(i) + ".png")
        # Have to swap columns because we ask for (line, column) instead of (x,y)
        depth_pixels[i][:, 0], depth_pixels[i][:, 1] = depth_pixels[i][:, 1], depth_pixels[i][:, 0].copy()
        distances[i] = image[tuple(depth_pixels[i].transpose().reshape(2, -1))][:, 0]
    return distances * rgb_camera_properties["scale"]
