import numpy as np

from api import get_facial_landmarks, optical_flow, color_to_depth_pixels, depth_pixels_to_distances

# # creating video based on images for optical flow
# from api import create_video_from_images
image_folder = '../inputs'
# video_name = 'video.avi'
# create_video_from_images(image_folder, video_name)

# face landmark detection for optical flow
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
image_path = "../inputs/0-0.png"
landmarks = get_facial_landmarks(shape_predictor_path, image_path, visualization=False)

# split the landmarks into different anatomical categories
landmarks_split = np.split(landmarks, [17, 22, 27, 36, 42, 48])
landmarks_name = ["contour", "right_eyebrow", "left_eyebrow", "nose", "right_eye", "left_eye", "mouth"]
landmarks_dictionary = {}
for i in range(len(landmarks_name)):
    landmarks_dictionary[landmarks_name[i]] = landmarks_split[i]

# optical flow for tracking the landmarks into the different images
video = "video.avi"
color_pixels = optical_flow(video, [landmarks_dictionary["right_eye"]], visualization=False)

# convert the pixels from the rgb image to depth image using pin hole model
json_path = "../info.json"
depth_pixels, rgb_camera_properties = color_to_depth_pixels(color_pixels, json_path)

# convert depth pixels to distances
distances = depth_pixels_to_distances(depth_pixels, image_folder, rgb_camera_properties)
