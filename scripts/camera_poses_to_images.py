#Script to process camera poses to nerfstudio render images

# The process flow of this script goes as follows:


#To use:
#1. config_path: Path to the config file. [bold red] REQUIRED


#Example usage:



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
# import textwrap

#================================================================================================================================================================
#CONSTANTS:

help_text = """
This script will take the input camera poses .json file and render the images using NeRFstudio.
    
The process flow of this script goes as follows:
1. Check if NeRFstudio is installed and whether the config file exists. If the mode is camera-path, check if the camera path file exists.
2. Run the ns-render command to render the images.
    
To use:
1. cd to the project directory where the NeRF model has been trained.
2. Run the script using the command: python PATH/TO/THIS/SCRIPT/camera_poses_to_images.py --config_path PATH/TO/CONFIG/FILE
3. If '--mode camera_path', use --camera_path PATH/TO/CAMERA/PATH/FILE as well.

IF this script is not in the project directory, specify the path to the project directory using the --base_path argument.

Example usage:
python PATH/TO/THIS/SCRIPT/camera_poses_to_images.py --config_path PATH/TO/CONFIG/FILE --camera_path PATH/TO/CAMERA/PATH/FILE 
* NOTE: verbose is default set true, so that the process can be seen in the terminal.
* use | tee console.txt if you want to save terminal output to a file

"""

ns_cmd = "ns-render"

#================================================================================================================================================================
#UTILITY FUNCTIONS:

#find the time taken and return the time taken in the format hh:mm:ss:ms
def time_taken(start_time):
    time_taken = time.time() - start_time
    return str(timedelta(seconds=time_taken))

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
default_mode = "camera-path"
default_config_path = Path(f"./processed/processed/{default_mode}/config.yml") #REQUIRED
default_camera_path = Path("./processed/processed/camera_paths/camera_path.json") #REQUIRED if mode is camera_path
default_output_name = "output"
default_output_path = Path(f'./renders/{default_mode}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/{default_output_name}')
default_output_format = "images"
default_interpolation_steps = 10
default_frame_rate = 24
default_split = "train"
default_occlusions = False
default_extra_args = ""


#parser
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)


#parser arguments
parser.add_argument("--verbose", dest="verbose", default=default_verbose, help=f"Print verbose output. \nDefault: {default_verbose}.", action="store_false")
parser.add_argument("--base_path", dest="base_path", type=Path, default=Path(default_base_path), help=f"Path to the project directory. \nDefault: {default_base_path}.")
parser.add_argument("--mode", dest="mode", type=str, default=default_mode, help=f"Mode of rendering. \nDefault: {default_mode}.")
parser.add_argument("--config_path", dest="config_path", type=Path, default=default_config_path, help=f"Path to the config file. \n REQUIRED: NO DEFAULT.")
parser.add_argument("--camera_path", dest="camera_path", type=Path, default=default_camera_path, help=f"Path to the camera path file. \nDefault: {default_camera_path}.")
parser.add_argument("--output_name", dest="output_name", type=str, default=default_output_name, help=f"Name of the output. \nDefault: {default_output_name}.")
parser.add_argument("--output_path", dest="output_path", type=Path, default=default_output_path, help=f"Path to save the output. \nDefault: {default_output_path}.")
parser.add_argument("--output_format", dest="output_format", type=str, default=default_output_format, help=f"Output format. \nDefault: {default_output_format}.")
parser.add_argument("--interpolation_steps", dest="interpolation_steps", type=int, default=default_interpolation_steps, help=f"Number of interpolation steps. \nDefault: {default_interpolation_steps}.")
parser.add_argument("--frame_rate", dest="frame_rate", type=int, default=default_frame_rate, help=f"Frame rate of the output video. \nDefault: {default_frame_rate}.")
parser.add_argument("--split", dest="split", type=str, default=default_split, help=f"Split of the dataset. \nDefault: {default_split}.")
parser.add_argument("--occlusions", dest="occlusions", default=default_occlusions, help=f"Check occlusions. \nDefault: {default_occlusions}.", action="store_true")
parser.add_argument("--extra_args", dest="extra_args", type=str, default=default_extra_args, help="Additional arguments for the ns-render command.")


#parse arguments
args=parser.parse_args()
verbose = args.verbose
base_path = Path(args.base_path)
mode = args.mode
config_path = Path(args.config_path)
camera_path = Path(args.camera_path)
output_name = args.output_name if args.output_name != default_output_name else f"{config_path.stem}_output"
output_path = Path(args.output_path) if args.output_path != default_output_path else Path(f'./renders/{default_mode}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/{output_name}')
output_format = args.output_format
interpolation_steps = args.interpolation_steps
frame_rate = args.frame_rate
split = args.split
occlusions = args.occlusions
extra_args = shlex.split(args.extra_args) if args.extra_args else []

#================================================================================================================================================================
#START OF SCRIPT:
#================================================================================================================================================================
#CHECKS:
    
CONSOLE.log("[bold white]Checking dependencies...")

#check if nerfstudio is installed
with status(msg="[bold yellow]Checking NeRFstudio installation...", spinner="moon"):
    try:
        import nerfstudio
    except ImportError:
        raise ImportError("NeRFstudio is not installed. Please install it from ...") #todo
    CONSOLE.log("[bold green]NeRFstudio is already installed.")
    
#check if ffmpeg is installed
with status(msg="[bold yellow]Checking ffmpeg installation...", spinner="moon"):
    try:
        run_command("ffmpeg -version")
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg is not installed. Please install it from ...") #todo
    CONSOLE.log("[bold green]ffmpeg is already installed.")
    
#check if config file exists
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}.")

#check if camera path file exists if mode is camera_path
if mode == "camera_path" and not camera_path.exists():
    raise FileNotFoundError(f"Camera path file not found at {camera_path}.")


CONSOLE.log("[bold green]\n:tada: All dependencies exist.\n",
            "[bold white]Starting Nerfstudio Rendering... (This will take a while +-45 mins)")

#================================================================================================================================================================
if mode == "camera-path":
    render_cmd = [
        str(ns_cmd), str(mode),
        "--load-config", str(config_path),
        "--camera-path-filename", str(camera_path),
        "--output-path", str(output_path),
        "--output-format", str(output_format),
        "--check-occlusions", str(occlusions),
    ] +extra_args
elif mode == "interpolate":
    render_cmd = [
        str(ns_cmd), str(mode),
        "--load-config", str(config_path),
        "--output-path", str(output_path),
        "--output-format", str(output_format),
        "--interpolation-steps", str(interpolation_steps),
        "--frame-rate", str(frame_rate),
        "--check-occlusions", str(occlusions),
    ] +extra_args
elif mode == "dataset":
    render_cmd = [
        str(ns_cmd), str(mode),
        "--load-config", str(config_path), 
        "--output-path", str(output_path),
        "--split", str(split),
        "--check-occlusions", str(occlusions),
    ] +extra_args
else:
    raise ValueError(f"Invalid mode: {mode}")

render_cmd = " ".join(render_cmd)
print(render_cmd)
start_time = time.time()

with status(msg="[bold yellow]Rendering... \n", spinner="moon", verbose=verbose):
    run_command(render_cmd, verbose=verbose)
    
time_render = time_taken(start_time)    
CONSOLE.log(f"[bold green]:tada: Rendering completed in {time_render}.","\n",
            f"[bold white]Output saved at: {output_path}.")


#================================================================================================================================================================
#FINISH SCRIPT
CONSOLE.log("[bold green]\n:tada: NeRFstudio Rendering Complete.")

#END OF SCRIPT