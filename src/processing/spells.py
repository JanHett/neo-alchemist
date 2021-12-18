import rawpy
import numpy as np

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
# COLOUR EDITING FUNCTIONS
####################################################

def white_balance(image, white_balance):
    """
    Balance image with given parameters
    """
    image[:,:,0] *= white_balance[0]
    image[:,:,1] *= white_balance[1]
    image[:,:,2] *= white_balance[2]

    return image

def gamma(image, gamma):
    """
    Adjust image contrast curve with `gamma` exponent
    """
    return np.maximum(0, image) ** gamma

def linear_contrast(image, shadows: float, highlights: float):
    """
    Adjust contrast linearly to parameters
    """
    slope = highlights - shadows

    return image * slope + shadows

def highlow_balance(image,
    shadow_red, shadow_green, shadow_blue,
    high_red, high_green, high_blue):
    """
    Adjust the balance of the highlights and shadows
    """
    shadows = np.array((shadow_red, shadow_green, shadow_blue))
    highs = np.array((high_red, high_green, high_blue))

    slope = highs - shadows
    return image * slope + shadows

def invert(image):
    """
    Invert the given image
    """

    return 1 - image
