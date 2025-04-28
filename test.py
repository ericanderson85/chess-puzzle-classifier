import os
import glob
import shutil

# List of source sub-directories
base_dirs = [
    "data/puzzles1/puzzles",
    "data/puzzles2/puzzles",
    "data/puzzles3/puzzles",
    "data/puzzles4/puzzles",
]

# Output directory
output_dir = "data/puzzles"
os.makedirs(output_dir, exist_ok=True)

# Collect all .pgn paths
filepaths = []
for bd in base_dirs:
    pattern = os.path.join(bd, "*.pgn")
    filepaths.extend(glob.glob(pattern))

filepaths.sort()

# Copy and rename
for idx, src in enumerate(filepaths):
    dst = os.path.join(output_dir, f"{idx}.pgn")
    shutil.copy(src, dst)

print(f"Copied {len(filepaths)} files into '{output_dir}/'")
