#script to rename the old images folder and take every nth image from the images folder and save it to images folder

#==============================================================================
#IMPORTS

import os
import shutil
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime, timedelta

#==============================================================================
#CONSTANTS

#==============================================================================
#UTILITY FUNCTIONS:

#find the time taken and return the time taken in the format hh:mm:ss:ms
def time_taken(start_time):
    time_taken = time.time() - start_time
    return str(timedelta(seconds=time_taken))

#==============================================================================
#ARGUMENTS

#defualt values
default_verbose = False
default_base_path = Path("./")
default_n = 2   
default_images_path = Path("images")
default_old_images_path = Path("original_images")

#parser
parser = argparse.ArgumentParser(description="script to rename the old images folder and take every nth image from the images folder and save it to images folder", formatter_class=argparse.RawTextHelpFormatter)

#parser arguments
parser.add_argument("--verbose", dest="verbose", default=default_verbose, help=f"Print verbose output. \nDefault: {default_verbose}.", action="store_true")
parser.add_argument("--base_path", dest="base_path", type=Path, default=Path(default_base_path), help=f"Path to the project directory. \nDefault: {default_base_path}.")
parser.add_argument("--n", dest="n", type=int, default=default_n, help=f"Take every nth image from the images folder. \nDefault: {default_n}.")
parser.add_argument("--images_path", dest="images_path", type=Path, default=default_images_path, help=f"Path to the images folder. \nDefault: {default_images_path}.")
parser.add_argument("--old_images_path", dest="old_images_path", type=Path, default=default_old_images_path, help=f"Path to the old images folder. \nDefault: {default_old_images_path}.")

#parser arguments
args=parser.parse_args()
verbose = args.verbose
base_path = Path(args.base_path)
n = args.n
images_path = Path(args.images_path)
old_images_path = Path(args.old_images_path)

#==============================================================================
#START OF SCRIPT

#check if the images folder exists
if not images_path.exists():
    print(f"Error: The images folder does not exist in the path {images_path}.")
    sys.exit(1)

#rename the images folder to old_images folder
if old_images_path.exists():
    raise Exception(f"Error: The old images folder already exists in the path {old_images_path}.")
else:
    os.rename(images_path, old_images_path)
    print(f"Renamed the images folder to old_images folder.")
    
#create a new images folder
os.makedirs(images_path)

start_time = time.time()

#take every nth image from the old images folder and save it to the images folder
old_images = sorted([str(old_images_path/image) for image in os.listdir(old_images_path) if image.endswith(".jpg") or image.endswith(".png")])

# total = 0
for i, old_image in enumerate(old_images):
    if i%n == 0:
        shutil.copy(old_image, images_path)
        if verbose:
            print(f"Copying {old_image} to {images_path}.")
        # total+=1

total_time = time_taken(start_time)

print(f"Completed copying every {n}th image from the old images folder to the images folder in {total_time}.", "\n","From total images:", len(old_images), "to total images:", len(os.listdir(images_path)))

#==============================================================================
#END OF SCRIPT


