import cv2
import os
import argparse
import re

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='/home/vhsiao1002/Data/colmap_nerf_data/images')
parser.add_argument('--output_folder', type=str, default='/home/vhsiao1002/Data/colmap_nerf_data/')
parser.add_argument('--frame_rate', type=int, default=30)
args = parser.parse_args()

# Define the path to the folder containing the images and the output video from parser
video_name = 'datasetVid.mp4'
image_folder = args.image_folder
output_video = os.path.join(args.output_folder, video_name)
frame_rate = args.frame_rate

# Get the list of images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]

# Function to extract numeric part from the filenames for sorting
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Sort images numerically based on the number extracted from filenames
images.sort(key=extract_number)

# Read the first image to get dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through the images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video_writer.write(frame)
    print(f"Writing frame {image}")

# Release the video writer object
video_writer.release()
cv2.destroyAllWindows()

print("\n---------------\nVideo created successfully!")