import os
from math import floor
from enum import Enum
from skimage import color
from typing import Tuple
import rawpy
import numpy as np
import numpy.typing as npt
from skimage.transform import resize
from numba import njit, prange

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
    return _white_balance_impl(image, np.array(white_balance, dtype=np.float32))

@njit(parallel=True)
def _white_balance_impl(image: ImageLike, white_balance: npt.ArrayLike):
    return image * white_balance

@njit(parallel=True)
def gamma(image: ImageLike, gamma: float):
    """
    Adjust image contrast curve with `gamma` exponent
    """
    return np.maximum(0, image) ** gamma

@njit(parallel=True)
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

    # shadow_pivot = 0.2
    # highlight_pivot = 0.8

    # base_len = highlight_pivot - shadow_pivot
    # high_low_diff = highs - shadow_pivot

    # slope = high_low_diff / base_len
    # return (image - shadow_pivot) * slope + shadows
    return _two_point_color_balance_impl(image, shadows, highs)

@njit(parallel=True)
def _two_point_color_balance_impl(image: ImageLike,
    shadow_balance: npt.ArrayLike, highlight_balance: npt.ArrayLike) -> ImageLike:
    return (image + shadow_balance) * highlight_balance

def cdl(image: ImageLike,
    slope: npt.ArrayLike,
    offset: npt.ArrayLike,
    power: npt.ArrayLike) -> ImageLike:
    return np.clip(image * slope + offset, 0, None) ** power

@njit(parallel=True)
def invert(image: ImageLike):
    """
    Invert the given image
    """

    return 1 - image

def interpolate_img(a: ImageLike, b: ImageLike, alpha: float) -> ImageLike:
    """
    Interpolates beween two images - an alpha of 0 yields image a, 1 yields b.
    Alpha values > 1 or < 0 extrapolate.
    """
    return (1 - alpha) * a + alpha * b

def saturation(image: ImageLike, saturation: float) -> ImageLike:
    """
    Adjust saturation by interpolation/extrapolation
    """
    bw = image.mean(axis=-1,keepdims=1)
    bw = np.repeat(bw, 3, axis=2)

    return interpolate_img(bw, image, saturation)

@njit(parallel=True)
def apply_color_transformation_matrix(image: ImageLike, mat: npt.ArrayLike):
    """
    Apply the color transformation matrix `mat`
    """
    for y in prange(image.shape[0]):
        for x in prange(image.shape[1]):
            image[y, x] = image[y, x] @ mat

def matrix_sat(image: ImageLike, saturation: float) -> ImageLike:
    """
    Adjust saturation with a colour matrix transform
    """
    # per-channel weights
    rwgt = 0.3086
    gwgt = 0.6094
    bwgt = 0.0820

    # matrix coefficients
    r_sat  = (1.0 - saturation) * rwgt + saturation
    r_coef = (1.0 - saturation) * rwgt
    g_sat  = (1.0 - saturation) * gwgt + saturation
    g_coef = (1.0 - saturation) * gwgt
    b_sat  = (1.0 - saturation) * bwgt + saturation
    b_coef = (1.0 - saturation) * bwgt

    mat = np.array((
        (r_sat,  r_coef, r_coef),
        (g_coef, g_sat,  g_coef),
        (b_coef, b_coef, b_sat)
    ), dtype=np.float32)

    ret = image @ mat

    return ret

def hue_sat(image: ImageLike, hue: float, saturation: float):
    """
    Modify hues and saturation by first transforming to the HSV space,
    multiplying the respective channels and then transforming back

    .. note::
        This method is relatively slow and will introduce some inaccuracies
    """
    hsv = color.rgb2hsv(image)

    hsv[:, :, 0] += hue
    hsv[:, :, 1] *= saturation

    return color.hsv2rgb(hsv)

####################################################
# COLOUR MANAGEMENT FUNCTIONS
####################################################

print("OCIO env variable:", os.environ["OCIO"])

class OCIOManager:
    def __init__(self):
        self.ocio_config = OCIO.GetCurrentConfig()

    def get_processor(self, from_space: str, to_space: str):
        proc = self.ocio_config.getProcessor(from_space, to_space)
        return proc.getDefaultCPUProcessor()

    def get_colorspaces(self):
        spaces = self.ocio_config.getColorSpaces()

        space_names = [s.getName() for s in spaces]

        return space_names

global_ocio = OCIOManager()
