# Conversion factor of X-ray pixels to mm.
# Estimated in notebooks/Pixel 2 mm.ipynb.
# Based on 10cm gantry height and 90° gantry angle, which is kept constant
# throughout all experiments.
PX2MM = 0.3214960197618965

# Sensor detection constants for YoloV5.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Default cropping window for X-ray.
DEFAULT_CROP = (79, 0, 557, 576)


def FT_to_DAP(ft: float) -> float:
    """
    Convert fluoroscopy time (FT) to dose area product (DAP).
    """
    return ft / 0.95 * 0.28

def DAP_to_FT(dap: float) -> float:
    """
    Convert dose area product (DAP) to fluoroscopy time (FT).
    """
    return dap / 0.28 * 0.95
