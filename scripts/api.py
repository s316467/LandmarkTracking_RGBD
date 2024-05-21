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
    # Initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rectangles = detector(gray, 1)

    if len(rectangles) == 0:
        if visualization:
            cv2.imshow("No faces detected", image)
            cv2.waitKey(0)
        raise ValueError("No faces detected")

    for (i, rect) in enumerate(rectangles):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if visualization:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    if visualization:
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        return shape



def optical_flow(image, landmarks=None, visualization=False, output_folder="output_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    frame_idx = 0
    saved_pixels = [p0.reshape(-1, 2)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
        img = cv2.add(frame, mask)

        # Save the current frame with the drawn tracks
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_idx:04d}.png"), img)
        frame_idx += 1

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        if not p0.shape[0] == good_new.shape[0]:
            assert False, "A tracked point was lost"
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()

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
    num_points, num_images = depth_pixels.shape[0], depth_pixels.shape[1]
    distances = np.empty((num_points, num_images))
    
    for i in range(3):
        image_path = os.path.join(image_folder, f"1-{i}.png")
        image = cv2.imread(image_path)
        
        # Check if the image is successfully loaded
        if image is None:
            raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
        
        # Ensure depth_pixels[i] has the expected dimensions
        if depth_pixels.shape[1] < 2:
            raise ValueError(f"Depth pixels array for image {i} does not have expected dimensions")
        
        # Swap columns because we ask for (line, column) instead of (x,y)
        depth_pixels[:, i, 0], depth_pixels[:, i, 1] = depth_pixels[:, i, 1], depth_pixels[:, i, 0].copy()
        distances[:, i] = image[tuple(depth_pixels[:, i].transpose().reshape(2, -1))][:, 0]
    
    return distances * rgb_camera_properties["scale"]

