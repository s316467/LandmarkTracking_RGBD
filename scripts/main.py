import cv2
import os
import numpy as np
from api import get_facial_landmarks, optical_flow, color_to_depth_pixels, depth_pixels_to_distances

# Creating video based on images for optical flow
image_folder = '../inputs'
video_name = 'video.avi'

# Face landmark detection for optical flow
shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"
image_path = "../inputs/0-0.png"
landmarks = get_facial_landmarks(shape_predictor_path, image_path, visualization=False)

# Split the landmarks into different anatomical categories
landmarks_split = np.split(landmarks, [17, 22, 27, 36, 42, 48])
landmarks_name = ["contour", "right_eyebrow", "left_eyebrow", "nose", "right_eye", "left_eye", "mouth"]
landmarks_dictionary = {}
for i in range(len(landmarks_name)):
    landmarks_dictionary[landmarks_name[i]] = landmarks_split[i]

# Optical flow for tracking the landmarks into the different images
output_folder = "output_images"
color_pixels = optical_flow(video_name, [landmarks_dictionary["right_eye"]], visualization=False, output_folder=output_folder)

# Convert the pixels from the RGB image to depth image using pin hole model
json_path = "../info.json"
depth_pixels, rgb_camera_properties = color_to_depth_pixels(color_pixels, json_path)

# Convert depth pixels to distances
distances = depth_pixels_to_distances(depth_pixels, image_folder, rgb_camera_properties)
print(distances)

# Display the saved images with tracked points
for filename in sorted(os.listdir(output_folder)):
    img = cv2.imread(os.path.join(output_folder, filename))
    cv2.imshow('Tracked Landmarks', img)
    if cv2.waitKey(500) & 0xFF == 27:  # Press 'ESC' to break the loop
        break

cv2.destroyAllWindows()
