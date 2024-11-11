# This script will process the images using glomap and then process the data for NeRFstudio training.

# The process flow of this script goes as follows:
#1. Check if the necessary dependencies exist (COLMAP, GLOMAP, NeRFstudio, database.db, images)
#2. Run COLMAP feature extractor
#3. Run COLMAP feature matcher
#4. Run GLOMAP mapper
#5. Run NeRFstudio process data
#6. Run NeRFstudio train


#To use:
#1. Have a project folder containing the images folder.
#2. cd into the project directory.
#3. Run the script using the command: python /PATH/TO/THIS/SCRIPT/glomap_nerf_process.py | tee console.txt

#IF this script is not in the project directory, specify the path to the project directory using the --base_path argument.

#Example usage:
#python /PATH/TO/THIS/SCRIPT/glomap_nerf_process.py --base_path /PATH/TO/PROJECT/DIRECTORY | tee console.txt


#================================================================================================================================================================
#IMPORTS:

from pathlib import Path
import appdirs
import numpy as np
import requests
from rich.progress import track
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command
import argparse
import time
import subprocess
import shlex
import sys
import os
import re
from datetime import datetime, timedelta

#================================================================================================================================================================
#CONSTANTS:
        
help_text = f"""
This script will process the images using glomap and then process the data for NeRFstudio training.

The process flow of this script goes as follows:
1. Check if the necessary dependencies exist (COLMAP, GLOMAP, NeRFstudio, database.db, images)
2. Run COLMAP feature extractor
3. Run COLMAP feature matcher
4. Run GLOMAP mapper
5. Run NeRFstudio process data
6. Run NeRFstudio train

To use:
1. Have a project folder containing the images folder.
2. cd into the project directory.
3. Run the script using the command: python /PATH/TO/THIS/SCRIPT/glomap_nerf_process.py | tee console.txt

IF this script is not in the project directory, specify the path to the project directory using the --base_path argument.

Example usage:
python /PATH/TO/THIS/SCRIPT/glomap_nerf_process.py --base_path /PATH/TO/PROJECT/DIRECTORY
* NOTE: verbose is default set true, so that the process can be seen in the terminal. 
* use | tee console.txt if you want to save terminal output to a file

"""
    
    
colmap_cmd = "colmap"
glomap_cmd = "glomap"

#================================================================================================================================================================
#UTIL FUNCTIONS:

#find time taken, if time more than 60s, convert to minutes, and return time taken and whether in seconds or minutes
def time_taken_convert(start_time):
    time_taken = time.time() - start_time
    if time_taken > 60:
        time_taken = time_taken / 60
        return f"{time_taken:.4f} mins", "mins"
    return f"{time_taken:.4f} s", "s"

#find the time taken and return the time taken in the format hh:mm:ss:ms
def time_taken(start_time):
    time_taken = time.time() - start_time
    return str(timedelta(seconds=time_taken))

#get vocab tree (from nerfstudio.process_data import colmap_utils)
def get_vocab_tree() -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfstudio")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename   


class TeeOutput:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = open(self.filename, "w")
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout

    def write(self, data):
        # Remove ANSI escape codes and other control characters
        # data = re.sub(r'(\x1B[@-_]|[\x80-\x9F]|[\x9B-\x9F]|[\xA0-\xFF])', '', data)
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

#================================================================================================================================================================
#ARGUMENTS:

#default values
default_verbose = True
default_base_path = Path("./")
default_image_path = Path(f"{default_base_path /'images'}")
default_database_path = Path(f"{default_base_path / 'database.db'}")
default_no_glomap = False
default_camera_model = "OPENCV"
default_camera_params = '529.4046769303875,529.6521348953188,647.1984132364281,354.6428014457265,0.007903972094598234,-0.002969179169396172,6.960621857680132e-05,-0.0012077776793605057' #default camera parameters for the blackfly camera
default_matching_method = "sequential"
default_use_gpu = 1
default_sparse_path = Path(f"{default_base_path / 'sparse'}")
default_nerf_method = "nerfacto"
default_depth_path = Path(f"{default_base_path / 'depth'}")
default_downscale_factor = 1
default_check_cuda = False
default_skip_extractor = False
default_skip_matcher = False
default_skip_mapper = False
default_skip_process_data = False
default_skip_train = False
default_feature_extractor_args = ""
default_feature_matcher_args = ""
default_glomap_mapper_args = ""
default_process_data_args = ""
default_train_args = ""

#parser  
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)


#parser arguments
parser.add_argument("--verbose", dest="verbose", default=default_verbose, help=f"Print verbose output. \nDefault: {default_verbose}.", action="store_false")
parser.add_argument("--base_path", dest="base_path", type=Path, default=Path(default_base_path), help=f"Path to the project directory. \nDefault: {default_base_path}.")
parser.add_argument("--image_path", dest="image_path", type=Path, default=Path(default_image_path), help=f"Path to the images. \nDefault: {default_image_path}.")
parser.add_argument("--database_path",dest="database_path",  type=Path, default=Path(default_database_path), help=f"Path to the database (including /database.db). \nDefault: {default_database_path}.")
parser.add_argument("--no_glomap", dest="no_glomap", default=default_no_glomap, help=f"Use GLOMAP for mapping. \nDefault: {default_no_glomap}", action="store_true")
parser.add_argument("--camera_model",dest="camera_model",  type=str, default=default_camera_model, help=f"Camera model (PINHOLE | OPENCV | RADIAL | SIMPLE_RADIAL | FISHEYE | SIMPLE_RADIAL_FISHEYE | OPENCV_FISHEYE). \nDefault: {default_camera_model}.")
parser.add_argument("--camera_params",dest="camera_params",  type=str, default=default_camera_params, help=f"Camera parameters (fx,fy,cx,cy,k1,k2,p1,p2). \nDefault: {default_camera_params}.")
parser.add_argument("--matching_method",dest="matching_method",  type=str, default=default_matching_method, help=f"COLMAP feature matcher method (exhaustive | sequential | vocab_tree ). \nDefault: {default_matching_method}.")
parser.add_argument("--use_gpu", dest="use_gpu", type=int, default=default_use_gpu, help=f"Use GPU for feature matching. \nDefault: {default_use_gpu}.")
parser.add_argument("--sparse_path", dest="sparse_path", type=Path, default=Path(default_sparse_path), help=f"Path to the sparse folder. \nDefault: {default_sparse_path}.")
parser.add_argument("--nerf_method", dest="nerf_method", type=str, default=default_nerf_method, help=f"NeRF method to train (nerfacto | depth-nerfacto | instant-ngp | splatfacto). \nDefault: {default_nerf_method}.")
parser.add_argument("--depth_path", dest="depth_path", type=Path, default=Path(default_depth_path), help=f"Path to the depth images. \nDefault: {default_depth_path}.")
parser.add_argument("--downscale_factor", dest="downscale_factor", type=int, default=default_downscale_factor, help=f"Downscale factor for depth-nerfacto method. \nDefault: {default_downscale_factor}.")
parser.add_argument("--check_cuda", dest="check_cuda", default=default_check_cuda, help=f"Check CUDA installation. \nDefault: {default_check_cuda}", action="store_true")
parser.add_argument("--skip_extractor", dest="skip_extractor", default=default_skip_extractor, help=f"Skip feature extractor. \nDefault: {default_skip_extractor}", action="store_true")
parser.add_argument("--skip_matcher", dest="skip_matcher", default=default_skip_matcher, help=f"Skip feature matcher. \nDefault: {default_skip_matcher}", action="store_true")
parser.add_argument("--skip_mapper", dest="skip_mapper", default=default_skip_mapper, help=f"Skip glomap mapper. \nDefault: {default_skip_mapper}", action="store_true")
parser.add_argument("--skip_process_data", dest="skip_process_data", default=default_skip_process_data, help=f"Skip NeRFstudio process data. \nDefault: {default_skip_process_data}", action="store_true")
parser.add_argument("--skip_train", dest="skip_train", default=default_skip_train, help=f"Skip NeRFstudio train. \nDefault: {default_skip_train}", action="store_true")
parser.add_argument("--extractor_args", dest="extractor_args", type=str, default=default_feature_extractor_args, help="Additional arguments for feature_extractor")
parser.add_argument("--matcher_args", dest="matcher_args", type=str, default=default_feature_matcher_args, help="Additional arguments for feature matcher")
parser.add_argument("--mapper_args", dest="mapper_args", type=str, default=default_glomap_mapper_args, help="Additional arguments for glomap mapper")
parser.add_argument("--process_data_args", dest="process_data_args", type=str, default=default_process_data_args, help="Additional arguments for NeRFstudio process data")
parser.add_argument("--train_args", dest="train_args", type=str, default=default_train_args, help="Additional arguments for NeRFstudio train")


#parse arguments
args=parser.parse_args()
verbose = args.verbose
base_path = Path(args.base_path)
image_path = Path(args.image_path) if args.image_path != default_image_path else Path(f"{base_path / 'images'}")
database_path = Path(args.database_path) if args.database_path != default_database_path else Path(f"{base_path / 'database.db'}")
no_glomap = args.no_glomap
camera_model = args.camera_model
camera_params = args.camera_params
matching_method = args.matching_method
use_gpu = args.use_gpu
sparse_path = Path(args.sparse_path) if args.sparse_path != default_sparse_path else Path(f"{base_path / 'sparse'}")
nerf_method = args.nerf_method
depth_path = Path(args.depth_path) if args.depth_path != default_depth_path else Path(f"{base_path / 'depth'}")
downscale_factor = args.downscale_factor
check_cuda = args.check_cuda
skip_extractor = args.skip_extractor
skip_matcher = args.skip_matcher
skip_mapper = args.skip_mapper
skip_process_data = args.skip_process_data
skip_train = args.skip_train
extractor_args = shlex.split(args.extractor_args) if args.extractor_args else []
matcher_args = shlex.split(args.matcher_args) if args.matcher_args else []
mapper_args = shlex.split(args.mapper_args) if args.mapper_args else []
process_data_args = shlex.split(args.process_data_args) if args.process_data_args else []
train_args = shlex.split(args.train_args) if args.train_args else []

             
#================================================================================================================================================================
#MAIN SCRIPT:

#console file name is current date and time
# console_folder = base_path / 'console'
# console_folder.mkdir(parents=True, exist_ok=True)
# console_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_console.txt"
# console_path = console_folder / console_filename
#use '| tee console.txt' to save console output to a file
#uncomment TeeOutput and tab in the script till process_data to save console output to a file.
# with TeeOutput(console_path):

#================================================================================================================================================================
#START OF SCRIPT:
#================================================================================================================================================================
#CHECKS:

CONSOLE.log("[bold white]Checking dependencies...")

#check CUDA installation (nvcc, available CUDA devices and CUDA device count) #but do not raise error if not installed
if (check_cuda):
    with status(msg="[bold yellow]Checking CUDA installation...", spinner="moon"):
        try:
            run_command("nvcc --version", verbose=verbose)
            run_command("nvidia-smi", verbose=verbose)
            import torch
            print(torch.cuda.is_available())
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        except FileNotFoundError:
            raise Exception("CUDA is not installed properly. Some features may not work.")
        else:
            CONSOLE.log("[bold green]CUDA is installed correctly.")

#check if colmap is installed
with status(msg="[bold yellow]Checking COLMAP installation...", spinner="moon"):
    try:
        run_command(f"{colmap_cmd} help", verbose=False)
    except FileNotFoundError:
        raise FileNotFoundError("COLMAP is not installed. Please install it from https://colmap.github.io/.")
    CONSOLE.log("[bold green]COLMAP is already installed.")

#check if glomap is installed
with status(msg="[bold yellow]Checking GLOMAP installation...", spinner="moon"):
    try:
        run_command(f"{glomap_cmd} --help", verbose=False)
    except FileNotFoundError:
        raise FileNotFoundError("GLOMAP is not installed. Please install it from ...") #todo
    CONSOLE.log("[bold green]Glomap is already installed.")

#check if nerfstudio is installed
with status(msg="[bold yellow]Checking NeRFstudio installation...", spinner="moon"):
    try:
        import nerfstudio
    except ImportError:
        raise ImportError("NeRFstudio is not installed. Please install it from ...") #todo
    CONSOLE.log("[bold green]NeRFstudio is already installed.")
    
#check if database exists and state it exists else create it at the specified path using the command: sqlite3 database.db "VACUUM;" 
if not database_path.exists():
    with status(msg="[bold yellow]database.db does not exist. Creating database...", spinner="moon"):
        run_command(f"sqlite3 {database_path} 'VACUUM;'", verbose=False)
    CONSOLE.log(f"[bold green]:tada: Database created at {database_path}.")
else:
    CONSOLE.log(f"[bold green]Database exists.")
    
#check if image_path exists and state number of images that are .jpg or .png
if not image_path.exists():
    raise FileNotFoundError(f"Image path {image_path} does not exist.")
else:
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_path}.")
    CONSOLE.log(f"[bold green]:tada: Found {len(image_files)} images in {image_path}.")
    

# CONSOLE.log(base_path)
# CONSOLE.log(image_path)
# CONSOLE.log(database_path)


CONSOLE.log("[bold green]\n:tada: All dependencies exist.\n",
            "[bold white]Starting GLOMAP Processing...")
    
#================================================================================================================================================================
#FEATURE EXTRACTOR

if not skip_extractor:
    feature_extractor_cmd = [
            str(colmap_cmd), "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", str(camera_model),
            "--ImageReader.camera_params", str(camera_params),
        ]  +extractor_args 
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    print(feature_extractor_cmd)

    #run feature extractor command 
    start_time = time.time()
    with status(msg="[bold yellow]Running COLMAP feature extractor... \n", spinner="runner"):
        run_command(feature_extractor_cmd, verbose=verbose)
    time_feature_extractor = time_taken(start_time)
    CONSOLE.log(f"[bold green]:tada: Done extracting COLMAP features. ({time_feature_extractor})")
else:
    CONSOLE.log("[bold yellow]Skipping feature extractor.")

#================================================================================================================================================================
#FEATURE MATCHER

if not skip_matcher:
    feature_matcher_cmd = [
            str(colmap_cmd), f"{matching_method}_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", str(use_gpu),
        ] +matcher_args
    if matching_method == "vocab_tree":
        vocab_tree_filename = get_vocab_tree()  
        feature_matcher_cmd.append(f'--VocabTreeMatching.vocab_tree_path "{vocab_tree_filename}"')
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    print(feature_matcher_cmd)

    #run feature matcher command
    start_time = time.time()
    with status(msg="[bold yellow]Running COLMAP feature matcher... \n", spinner="runner", verbose=verbose):
        run_command(feature_matcher_cmd, verbose=verbose)
    time_feature_matcher = time_taken(start_time)
    CONSOLE.log(f"[bold green]:tada: Done matching COLMAP features. ({time_feature_matcher})")
else:
    CONSOLE.log("[bold yellow]Skipping feature matcher.")

#================================================================================================================================================================
#MAPPER

sparse_path.mkdir(parents=True, exist_ok=True)

if no_glomap:
    glomap_cmd = "colmap"

if not skip_mapper:
    glomap_mapper_cmd = [
        str(glomap_cmd), "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_path),
    ] +mapper_args
    glomap_mapper_cmd = " ".join(glomap_mapper_cmd)
    print(glomap_mapper_cmd)

    #run glomap mapper command
    start_time = time.time()
    with status(msg="[bold yellow]Running GLOMAP mapper... (This may take a while) \n", spinner="runner", verbose=verbose):
        run_command(glomap_mapper_cmd, verbose=verbose)
    time_glomap_mapper = time_taken(start_time)
    CONSOLE.log(f"[bold green]:tada: Done mapping images with GLOMAP. ({time_glomap_mapper})","\n",
                "[bold white]View the glomap sparse model using:","\n",f"colmap gui --image_path {image_path} --database_path {database_path} --import_path {sparse_path / '0'}")
else:
    CONSOLE.log("[bold yellow]Skipping glomap mapper.")

#================================================================================================================================================================
CONSOLE.log("[bold green]\n:tada: GLOMAP Processing complete.\n",
            "[bold white]Starting NeRFstudio data processing...")

#================================================================================================================================================================
#NERFSTUDIO PROCESS DATA

if not skip_process_data:
    ns_process_data_cmd = [
        "ns-process-data images",
        "--data", str(image_path),
        "--output-dir", str(base_path / 'processed'),
        "--skip-colmap",
        "--colmap-model-path", str('..' / sparse_path / '0'),
    ] +process_data_args
    ns_process_data_cmd = " ".join(ns_process_data_cmd)
    print(ns_process_data_cmd)

    #run ns-process-data command
    start_time = time.time()
    with status(msg="[bold yellow]Running NeRFstudio process data... \n", spinner="runner", verbose=verbose):
        run_command(ns_process_data_cmd, verbose=verbose)
    time_ns_process_data = time_taken(start_time)
    CONSOLE.log(f"[bold green]:tada: Done processing data for NeRFstudio. ({time_ns_process_data})")
else:
    CONSOLE.log("[bold yellow]Skipping NeRFstudio process data.")

#================================================================================================================================================================
#NERFSTUDIO TRAIN

processed_path = base_path / "processed"

if(not skip_train):
    ns_train_cmd = ["ns-train", nerf_method]
    if nerf_method == "depth-nerfacto":
        ns_train_cmd.extend([
            "colmap",
            "--data", str(processed_path),
            "--colmap-path", str(sparse_path / '0'),
            "--load-3D-points", "True",
            "--images-path", str(image_path),
            "--depths-path", str(depth_path),
            "--downscale", "factor", str(downscale_factor),
            "--output-dir", str(processed_path),
        ]) #+train_args
    else:
        ns_train_cmd.extend([
            "--data", str(processed_path),
            "--output-dir", str(processed_path),
        ]) #+train_args   
        
    if train_args:
        ns_train_cmd.extend(train_args)
        
    ns_train_cmd = " ".join(ns_train_cmd)
    print(ns_train_cmd)
                     
         
    #run ns-train command
    start_time = time.time()
    
    # print(" ".join(ns_train_cmd))
    # process = subprocess.Popen(ns_train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # for line in process.stdout:
    #     print(line, end="")
    # process.wait()
    # if process.returncode != 0:
    #     error_output = process.stderr.read()
    #     print(f"Error: {error_output}")
    
    with status(msg="[bold yellow]Running NeRFstudio train... \n", spinner="runner", verbose=verbose):
        run_command(ns_train_cmd, verbose=verbose)
        
    time_ns_train = time_taken(start_time)
    CONSOLE.log(f"[bold green]:tada: Done {nerf_method} training. ({time_ns_train})","\n",
                f"[bold white]The config is saved at {base_path}/. Use ns-viewer --load-config path/to/config.yml to view the results ({processed_path}).")
else:
    CONSOLE.log("[bold yellow]Skipping NeRFstudio train.")


#================================================================================================================================================================
#FINISH SCRIPT
CONSOLE.log("[bold green]\n:tada: NeRFstudio Processing complete.\n",)

#END OF SCRIPT