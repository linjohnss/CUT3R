import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    # Get the src directory path by finding the directory that contains dust3r module
    ckpt_abs_path = os.path.abspath(ckpt)
    current_dir = os.path.dirname(ckpt_abs_path)
    
    # Walk up the directory tree to find the src directory (which contains dust3r)
    while current_dir != os.path.dirname(current_dir):  # Stop at root
        if os.path.exists(os.path.join(current_dir, "dust3r")):
            # Found the src directory
            sys.path.insert(0, current_dir)
            break
        current_dir = os.path.dirname(current_dir)
    else:
        # Fallback: add the directory containing the checkpoint
        HERE_PATH = os.path.dirname(ckpt_abs_path)
        sys.path.insert(0, HERE_PATH)
