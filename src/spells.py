import rawpy
import numpy as np

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

class ColorNegative:
    def __init__(self, img_path: str):
        self._data = rawpy.imread(img_path)

        self._linear_base = self._data.postprocess(
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            no_auto_bright=True
            )

    def linear_base(self):
        return self._linear_base

    def white_balanced(self, x1, x2, y1, y2):
        wb_patch = self._linear_base[y1:y2, x1:x2]

        avg_r = np.average(wb_patch[..., 0])
        avg_g = np.average(wb_patch[..., 1])
        avg_b = np.average(wb_patch[..., 2])

        base_wb = [avg_g/avg_r, 1.0, avg_g/avg_b, 1.0]

        return self._data.postprocess(user_wb=base_wb)

    def positive(self):
        balanced = self.white_balanced(50,300,2000,8500)

        max_r = np.max(balanced[..., 0])
        max_g = np.max(balanced[..., 1])
        max_b = np.max(balanced[..., 2])

        balanced[..., 0] = max_r - balanced[..., 0]
        balanced[..., 1] = max_g - balanced[..., 1]
        balanced[..., 2] = max_b - balanced[..., 2]

        return balanced

    def highlights(self, r, g, b):
        pass

    def shadows(self, r, g, b):
        pass

def white_balance(raw: rawpy.RawPy, sampling_region: tuple[int]):
    pass
