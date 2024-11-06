#python script for converting raw images and poses from the husky into nerfstudio format
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

########################################################################################################
#path to data
data_path = "/home/vhsiao1002/Data/"

#name of the dataset
dataset_name = "nerf_data"
########################################################################################################


#pytorch trnsformation to convert PIl image to tensor
transform = transforms.Compose([transforms.PILToTensor()])



#argument parser
parser = argparse.ArgumentParser(description="Preprocess husky data into nerfstudio format")

#default values
default_input_path = "/home/vhsiao1002/Data/nerf_data"
default_output_path = "/home/vhsiao1002/Data/nerf_data"
default_sampling_rate = 1

default_threshold=1000
default_initial_cutoff = 100
default_ratio = 10

#path to input rgb images
parser.add_argument("--input_path", dest="input_path", default=default_input_path)

#path to save processed images to
parser.add_argument("--output_path", dest="output_path", default=default_output_path)

#sampling rate at which to extract images
parser.add_argument("--sampling_rate", dest="sampling_rate", type=int, default=default_sampling_rate)

#threshold for number of images
parser.add_argument("--threshold", dest="threshold", type=int, default=default_threshold)

#initial cutoff for number of images
parser.add_argument("--initial_cutoff", dest="initial_cutoff", type=int, default=default_initial_cutoff)

#ratio for number of images
parser.add_argument("--ratio", dest="ratio", type=int, default=default_ratio)

args = parser.parse_args()

output_path = Path(args.output_path)
input_path = Path(args.input_path)
sampling_rate = args.sampling_rate
threshold = args.threshold
initial_cutoff = args.initial_cutoff
ratio = args.ratio

#create path to colour images
colour_path = input_path / "images"
#extract all the colour image paths in the folder
colour_paths = sorted(glob.glob(os.path.join(colour_path, "*.png")),
                      key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)[:-4]))))


#create path to pose files
pose_path = input_path / "poses"
poses=[]
#extract all the pose files in the folder
pose_paths = sorted(glob.glob(os.path.join(pose_path, "*.txt")),
                    key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)[:-4]))))
for pose_path in pose_paths:
    camera_to_world = np.loadtxt(pose_path)
    poses.append(camera_to_world)
poses = np.array(poses)

#creates path to depth images
depth_path = input_path / "depth"
#extract all the depth imae paths in the folder
depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")),
                     key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)[:-4]))))

#open one of the rgb images for reference to create the mask
example_image_path = colour_path / "image_2.png"
image_rgb = cv2.imread(str(example_image_path))

baseplate_mask = False
segmentation = True

pointcloud = True

frames = []

counter=0

for filename in os.listdir(colour_path):
    idx = filename.split(".")[0]
    idx_int = str(''.join(filter(str.isdigit, filename.split(".")[0])))
    # print(idx, int(idx_int))
    
    filename = str(idx) + ".png"
    mask_filename = "" +data_path +dataset_name+ "/segmentation/" + filename +".png"
    # mask_filename = "" +data_path +dataset_name+ "/segmentation/" + ''.join(filter(str.isdigit, filename.split(".")[0])) + ".png"
    # print(mask_filename)
    # num_segments = 288
    # num_images = 1152


    if (int(idx_int) % int(sampling_rate) != 0 or counter > threshold/ratio or int(idx_int) > threshold or int(idx_int) < initial_cutoff):
        continue
    else:
        # create grayscale mask
        if (os.path.isfile(mask_filename) and segmentation):
            filename = str(idx) + ".png"
            print(filename)
            mask_filename = "" +data_path+dataset_name+"/segmentation/" + filename
            mask = cv2.imread(mask_filename, cv2.COLOR_BGR2GRAY)

            # iterate through rgb image and create mask
            for i in range(720):
                for j in range(1280):
                    if (baseplate_mask):
                        if (i > 998 and (j < 1220 and j > 910)):
                            mask[i][j] = 0
                        else:
                            if (not (mask[i][j] == 0)):
                                mask[i][j] = 0
                            else:
                                mask[i][j] = 255
                    else:
                        if (not (mask[i][j] == 0)):
                            mask[i][j] = 0
                        else:
                            mask[i][j] = 255
                    # save masked image
            mask_path = "" +data_path+dataset_name+"/masks/mask_" + str(idx_int) +".png"
            cv2.imwrite(mask_path, mask)
        elif(segmentation == True):
            # mask_path = "" +data_path+dataset_name+"/baseplate_mask.png"
            
            #VH
            # mask_num = 
            # mask_path = "" +data_path+dataset_name+"/masks/mask_" + ''.join(filter(str.isdigit, filename.split(".")[0])) +".png"
            mask_path = "" +data_path+dataset_name+"/masks/mask_" + str(idx_int) +".png"
            
        else:
            mask_path = None



        #path to rgb image
        rgb_path = "" +data_path +dataset_name+"/images/"+filename

        #path to transforms file
        pose_path = "" +data_path +dataset_name+"/poses/transform_"+str(idx_int)+".txt"

        #extract poses in numpy array
        pose = np.loadtxt(pose_path)
        pose[0:3, 1:3] *= -1
        pose = pose[np.array([1, 0, 2, 3]), :]
        pose[2, :] *= -1

        

        #path to depth file
        depth_path = "" +data_path +dataset_name+"/depth/depth_"+str(idx_int)+".png"
        
        #path to point cloud file
        int_pcd = int(idx_int)*2
        point_cloud_path = "" +data_path +dataset_name+"/point_cloud/point_cloud_"+str(int_pcd)+".ply"
        # point_cloud_path = "" +data_path +dataset_name+"/velodyne_points/point_cloud_"+str(int_pcd)+".pcd"
        

        frame = {
            "file_path": rgb_path,
            "transform_matrix": pose.tolist(),
            "depth_file_path": depth_path,
            "mask_path": mask_path,
            "ply_file_path": point_cloud_path,
        }


        frames.append(frame)

        counter = counter + 1
        print(f"{idx_int}\t{counter}")


# meta data
output_data = {
    "camera_model": "OPENCV",
    "fl_x" :  531.14774,
    "fl_y" : 531.26312,
    "cx" : 637.87114,
    "cy" : 331.27469,
    "w": 1280,
    "h": 720,
#     "k1" : -0.048045,
#     "k2" : 0.010493,
#     "k3" : 0.000000,
#     "p1" : -0.000281,
#     "p2" : -0.001232,
}

# output_data = {
#     "camera_model": "OPENCV",
#     "fl_x" :  525.03253174,
#     "fl_y" : 525.03253174,
#     "cx" : 648.5067749,
#     "cy" : 357.38909912,
#     "w": 1280,
#     "h": 720,
# }




# output_data = {
#     "camera_model": "pinhole",
#     "fl_x": 527.5591059906969,  # Focal length in x
#     "fl_y": 528.5624579927512,  # Focal length in y
#     "cx": 647.1975009993375,    # Principal point x-coordinate
#     "cy": 357.2476935284654,    # Principal point y-coordinate
#     "w": 1280,                  # Image width
#     "h": 720,                   # Image height
#     "k1": 0.004262406434905663, # First radial distortion coefficient
#     "k2": -0.030631455483041737,# Second radial distortion coefficient
#     "k3": 5.567440162484537e-05,# Third radial distortion coefficient
#     "p1": -0.00079751451332914, # First tangential distortion coefficient
#     "p2": 0.0  
# }

output_data["frames"] = frames

# output_data["ply_file_path"] = "/home/vhsiao1002/Data/nerf_data/point_cloud/point_cloud_96.ply"

# save as json
with open(output_path / "transforms.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)


















