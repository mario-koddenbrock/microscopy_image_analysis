import os
import warnings

import cellpose
import rasterio

# Ignore specific warnings
ignored_warnings = [
    rasterio.errors.NotGeoreferencedWarning,
    DeprecationWarning,
    FutureWarning,
    UserWarning
]

for warning in ignored_warnings:
    warnings.filterwarnings("ignore", category=warning)

# Set the path to the ffmpeg executable - only needed for exporting animations
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
print(f"Cellpose version: {cellpose.version}")
