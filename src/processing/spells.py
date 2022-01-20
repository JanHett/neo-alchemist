import os
from math import floor
from enum import Enum
from typing import Tuple
import rawpy
import numpy as np
import numpy.typing as npt
from skimage.transform import resize

import PyOpenColorIO as OCIO

####################################################
# TYPE ALIASES
####################################################

ImageLike = npt.ArrayLike
Color = Tuple[float, float, float]

####################################################
# SMOOTH STEPPING FUNCTIONS
####################################################

def smoothstart2(x):
    return x * x
def smoothstart3(x):
    return x * x * x
def smoothstart4(x):
    return x * x * x * x
def smoothstart5(x):
    return x * x * x * x * x
def smoothstart6(x):
    return x * x * x * x * x * x

def smoothstop2(x):
    x_inv = 1 - x
    return 1 - x_inv * x_inv

def smoothstop3(x):
    x_inv = 1 - x
    return 1 - x_inv * x_inv * x_inv

def smoothstop4(x):
    x_inv = 1 - x
    return 1 - x_inv * x_inv * x_inv * x_inv

def smoothstop5(x):
    x_inv = 1 - x
    return 1 - x_inv * x_inv * x_inv * x_inv * x_inv

def smoothstop6(x):
    x_inv = 1 - x
    return 1 - x_inv * x_inv * x_inv * x_inv * x_inv * x_inv

def mix(f1, f2, weight, x):
    return (1 - weight) * f1(x) + weight * f2(x)

def crossfade(f1, f2, x):
    return (1 - x) * f1(x) + x * f2(x)

def smoothstep3(x):
    return crossfade(smoothstart2, smoothstop2, x)

####################################################
# IMAGE FITTING FUNCTIONS
####################################################

class ImageFit(Enum):
    CONTAIN = "CONTAIN",
    COVER = "COVER"

def fit_image(img,
    resolution: tuple[int, int],
    fit: ImageFit = ImageFit.CONTAIN) -> np.ndarray:
    w = img.shape[0]
    h = img.shape[1]

    in_aspect = w/h
    out_aspect = resolution[0] / resolution[1]

    if fit is ImageFit.CONTAIN:
        # input is wider than output window
        if in_aspect > out_aspect:
            out_width = resolution[0]
            out_height = resolution[0] / in_aspect
        # input is taller or same aspect as output window
        else:
            out_height = resolution[1]
            out_width = resolution[1] * in_aspect

    elif fit is ImageFit.COVER:
        # input is wider than output window
        if in_aspect > out_aspect:
            out_width = resolution[0] / in_aspect
            out_height = resolution[0]
        # input is taller or same aspect as output window
        else:
            out_height = resolution[1] * in_aspect
            out_width = resolution[1]

    else:
        raise ValueError(f"Fit '{fit}' is not supprted")
    
    return resize(img, (out_width, out_height), anti_aliasing=True)

####################################################
# COLOUR EDITING FUNCTIONS
####################################################

def white_balance(image: ImageLike, white_balance: Tuple[float, float, float]):
    """
    Balance image with given parameters
    """
    image[:,:,0] *= white_balance[0]
    image[:,:,1] *= white_balance[1]
    image[:,:,2] *= white_balance[2]

    return image

def gamma(image: ImageLike, gamma: float):
    """
    Adjust image contrast curve with `gamma` exponent
    """
    return np.maximum(0, image) ** gamma

def linear_contrast(image: ImageLike, lift: float, slope: float):
    """
    Adjust contrast linearly to parameters
    """
    return image * slope + lift

def two_point_color_balance(image: ImageLike,
    shadow_balance: Color, highlight_balance: Color) -> ImageLike:
    """
    Adjust the balance of the highlights and shadows

    The shadows balance should revolve around 0.2, hoghlight around 0.8

    TODO: make shadow and highlight pivots dynamic instead of hardcoded
    """
    shadows = np.array(shadow_balance, dtype=np.float32)
    highs = np.array(highlight_balance, dtype=np.float32)

    shadow_pivot = 0.2
    highlight_pivot = 0.8

    base_len = highlight_pivot - shadow_pivot
    high_low_diff = highs - shadows

    slope = high_low_diff / base_len
    return (image - shadow_pivot) * slope + shadows

def invert(image: ImageLike):
    """
    Invert the given image
    """

    return 1 - image

####################################################
# COLOUR MANAGEMENT FUNCTIONS
####################################################

print("OCIO env variable:", os.environ["OCIO"])

ocio_config = OCIO.GetCurrentConfig()

ocio_display = ocio_config.getDefaultDisplay()
ocio_view = ocio_config.getDefaultView(ocio_display)
ocio_processor = ocio_config.getProcessor(
    OCIO.ROLE_SCENE_LINEAR,
    ocio_display,
    ocio_view,
    OCIO.TRANSFORM_DIR_FORWARD)
ocio_cpu = ocio_processor.getDefaultCPUProcessor()

def lin_to_display(image):
    """
    Applies a scene-linear to display colour space transform to a copy of the
    image
    """
    to_convert = image.copy()
    ocio_cpu.applyRGB(to_convert)

    print(f"max: {np.max(to_convert)}")
    print(f"min: {np.min(to_convert)}")

    return to_convert

srgb_processor = ocio_config.getProcessor(OCIO.ROLE_SCENE_LINEAR, "Output - sRGB")
srgb_cpu = srgb_processor.getDefaultCPUProcessor()

def lin_to_srgb(image):
    to_convert = image.copy()
    print(f"[[ BEFORE TRANSFORM ]] max: {np.max(to_convert)}")
    print(f"[[ BEFORE TRANSFORM ]] min: {np.min(to_convert)}")

    srgb_cpu.applyRGB(to_convert)

    print(f"[[ AFTER TRANSFORM ]] max: {np.max(to_convert)}")
    print(f"[[ AFTER TRANSFORM ]] min: {np.min(to_convert)}")
    print(f"to_convert.dtype: {to_convert.dtype}")

    return to_convert
