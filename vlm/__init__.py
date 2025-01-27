import warnings
import rasterio


# Ignore specific warnings
ignored_warnings = [
    rasterio.errors.NotGeoreferencedWarning,
    DeprecationWarning,
    FutureWarning,
    UserWarning,
]

for warning in ignored_warnings:
    warnings.filterwarnings("ignore", category=warning)
