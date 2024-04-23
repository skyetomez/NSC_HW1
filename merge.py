import os
import subprocess

# Get a list of all .ipynb files in the current directory
notebook_files = [f for f in os.listdir() if f.endswith('.ipynb')]

# Sort the files to concatenate them in order
notebook_files.sort()

# Use nbmerge to concatenate the files
subprocess.run(['nbmerge'] + notebook_files + ['-o',
               '/home/skyler/Documents/SP2024/580_695/homeworks/NSC_HW1/sthom215.ipynb'])
