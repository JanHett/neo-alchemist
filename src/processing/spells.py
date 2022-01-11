from math import floor
from enum import Enum
from typing import Tuple
import rawpy
import numpy as np
import numpy.typing as npt
from skimage.transform import resize

####################################################
# TYPE ALIASES
####################################################

ImageLike = npt.ArrayLike

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

def linear_contrast(image: ImageLike, shadows: float, highlights: float):
    """
    Adjust contrast linearly to parameters
    """
    slope = highlights - shadows

    return image * slope + shadows

def highlow_balance(image: ImageLike,
    shadow_red: float, shadow_green: float, shadow_blue: float,
    high_red: float, high_green: float, high_blue: float):
    """
    Adjust the balance of the highlights and shadows
    """
    shadows = np.array((shadow_red, shadow_green, shadow_blue))
    highs = np.array((high_red, high_green, high_blue))

    slope = highs - shadows
    return image * slope + shadows

def invert(image: ImageLike):
    """
    Invert the given image
    """

    return 1 - image
