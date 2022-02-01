from typing import Optional, Union
from PySide2.QtCore import QRect, Qt, Signal, QPoint
from PySide2.QtWidgets import \
    QGridLayout, \
    QLabel, \
    QOpenGLWidget, \
    QDoubleSpinBox, \
    QVBoxLayout, \
    QWidget
# from PySide2.QtOpenGLWidgets import QOpenGLWidget
from PySide2.QtGui import QColor, QImage, QPaintEvent, QPainter, QPainterPath, QMouseEvent
from superqt import QDoubleSlider

from math import pi

import numpy as np
import numpy.typing as npt

from colorio.cs import ColorCoordinates, CIELCH, XYZ

from numba import prange, njit, float32

from ...processing.spells import ImageLike, Color, global_ocio

class LabelledSlider(QWidget):
    def __init__(self, label: str, parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, f=f)

        self.label = QLabel(label, self)
        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.spinbox.setDecimals(4)
        self.slider = QDoubleSlider(Qt.Horizontal, self)

        self.slider.valueChanged.connect(self._update_spinbox)
        self.spinbox.editingFinished.connect(self._update_slider)

        self._layout = QGridLayout(self)
        self._layout.setColumnStretch(0, 2)
        self._layout.setColumnStretch(1, 1)
        self._layout.addWidget(self.label, 0, 0)
        self._layout.addWidget(self.spinbox, 0, 1)
        self._layout.addWidget(self.slider, 1, 0, 1, 2)

        self.setLayout(self._layout)

        self.valueChanged = self.slider.valueChanged

    def _update_spinbox(self):
        self.spinbox.setValue(self.slider.value())

    def _update_slider(self):
        self.slider.setValue(self.spinbox.value())

    def setValue(self, v: float) -> None:
        self.slider.setValue(v)
        self.spinbox.setValue(v)

    def setMaximum(self, v: float) -> None:
        self.slider.setMaximum(v)
        self.spinbox.setMaximum(v)

    def setMinimum(self, v: float) -> None:
        self.slider.setMinimum(v)
        self.spinbox.setMinimum(v)

    def value(self) -> float:
        return self.slider.value()

class ViewerWidget(QWidget):
    def __init__(self,
        parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent, f)

        self._layout = QVBoxLayout(self)
        self._renderer = ImageRenderer(np.zeros((1,1,3), dtype=np.float32))
        self._layout.addWidget(self._renderer)
        self.setLayout(self._layout)

    @property
    def image_renderer(self):
        return self._renderer

class ColorPicker(QWidget):
    """
    A general purpose color picker
    """

    color_changed = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent, f)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.red = LabelledSlider("Red", self)
        self.green = LabelledSlider("Green", self)
        self.blue = LabelledSlider("Blue", self)

        self.setMinimum(0)
        self.setMaximum(1)
        self.setValue(1)

        self.red.valueChanged.connect(self._emit_color_changed)
        self.green.valueChanged.connect(self._emit_color_changed)
        self.blue.valueChanged.connect(self._emit_color_changed)

        self._layout.addWidget(self.red)
        self._layout.addWidget(self.green)
        self._layout.addWidget(self.blue)

        self.setLayout(self._layout)

    def setMinimum(self, value: Union[Color, float]) -> None:
        """
        Set minimum for all channels

        If a `Color` is provided, its channels are taken to refer to RGB values
        """
        try:
            self.red.setMinimum(value[0])
            self.green.setMinimum(value[1])
            self.blue.setMinimum(value[2])
        except TypeError:
            self.red.setMinimum(value)
            self.green.setMinimum(value)
            self.blue.setMinimum(value)

    def setMaximum(self, value: Union[Color, float]) -> None:
        """
        Set maximum for all channels

        If a `Color` is provided, its channels are taken to refer to RGB values
        """
        try:
            self.red.setMaximum(value[0])
            self.green.setMaximum(value[1])
            self.blue.setMaximum(value[2])
        except TypeError:
            self.red.setMaximum(value)
            self.green.setMaximum(value)
            self.blue.setMaximum(value)

    def setValue(self, value: Union[Color, float]) -> None:
        try:
            self.red.setValue(value[0])
            self.green.setValue(value[1])
            self.blue.setValue(value[2])
        except TypeError:
            self.red.setValue(value)
            self.green.setValue(value)
            self.blue.setValue(value)

    def value(self) -> Color:
        return (self.red.value(), self.green.value(), self.blue.value())

    def _emit_color_changed(self, _):
        self.color_changed.emit(self.value())

class ImageRenderer(QOpenGLWidget):
    """
    Displays an image on screen
    """
    def __init__(self,
        image: np.ndarray,
        parent: Optional[QWidget] = None,
        f: Qt.WindowFlags = Qt.WindowFlags()
        ) -> None:
        super().__init__(parent=parent, f=f)
        self.set_image(image)

        self._overlay = []

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def paintEvent(self, event: QPaintEvent):
        img = image_to_QImage(self.image)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # draw the image
        painter.drawImage(0, 0, img)
        # draw overlays
        painter.setPen(QColor(0, 255, 0))
        painter.drawRects(self._overlay)

    @property
    def overlay(self):
        return self.overlay
    
    @overlay.setter
    def overlay(self, overlay: list[QRect]):
        self._overlay = overlay

    def set_image(self, image: ImageLike):
        self.image = image
        self.setFixedSize(self.image.shape[1], self.image.shape[0])

D60xy = [0.321616709705268, 0.337619916550817]
D60 = [D60xy[0] / D60xy[1] * 100, 1 * 100, (1 - D60xy[0] - D60xy[1]) / D60xy[1] * 100]
class ColorBalanceControl(QOpenGLWidget):

    value_changed = Signal(tuple)

    def __init__(self,
        # neutral_value = 1,
        parent: Optional[QWidget] = None,
        f: Qt.WindowFlags = Qt.WindowFlags()):
        super().__init__(parent, f)
        self._ocio_proc = global_ocio.get_processor("Utility - XYZ - D60", "Output - P3D65")
        # self._ocio_proc = global_ocio.get_processor("Role - scene_linear", "Output - P3D65")
        self._xyz_wheel = self._generate_wheel()
        self._p3_wheel = self._xyz_wheel.copy()
        self._ocio_proc.applyRGB(self._p3_wheel)
        self.setFixedSize(self.width, self.height)

        self._previous_pos = QPoint(0, 0)
        self._pos = np.array((0.5, 0.5))

        self._factor = 0.1

    def paintEvent(self, event: QPaintEvent):
        w = self._p3_wheel
        self.setFixedSize(self.width, self.height)
        wheel = image_to_QImage(w)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # draw the image inside a circle
        clip_elipse = QPainterPath()
        clip_elipse.addEllipse(0, 0, self.width, self.height)
        painter.setClipPath(clip_elipse)
        painter.drawImage(0, 0, wheel)
        # draw overlays
        painter.setPen(QColor(0, 255, 0))
        marker_y, marker_x = (self._pos * self._xyz_wheel.shape[:2]).astype(np.int)
        radius = 10
        painter.drawEllipse(marker_x - radius, marker_y - radius, 2 * radius, 2 * radius)
        # painter.drawRects(self._overlay)

    def mousePressEvent(self, event: QMouseEvent):
        self._previous_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        delta = event.pos() - self._previous_pos
        delta = np.array((delta.y(), delta.x())) / self._xyz_wheel.shape[:2]
        # print("delta:", delta)
        self._previous_pos = event.pos()
        self._pos = self._pos + delta * self._factor
        # print("self._pos:", self._pos)
        # print("self.xyz_color:", self.xyz_color)
        # print("self._p3_color:", self._p3_color)
        self.value_changed.emit(self.xyz_color)
        self.update()


    @property
    def xyz_color(self):
        y, x = (self._pos * self._xyz_wheel.shape[:2]).astype(np.int)
        return self._xyz_wheel[y, x]

    @property
    def _p3_color(self):
        c = self.xyz_color.copy()
        self._ocio_proc.applyRGB(c)
        return c

    def get_value(self):
        return self.xyz_color

    def set_value(self, value):
        # TODO
        pass

    def _generate_wheel(self):
        w = np.zeros((self.height, self.width, 3), dtype=np.float32)
        # draw_color_wheel(w)
        draw_color_wheel_lch(w)
        w_moved = np.moveaxis(w, 2, 0)
        c_coords = ColorCoordinates(w_moved, CIELCH(whitepoint=D60))
        c_coords.convert(XYZ(1))
        w = np.moveaxis(c_coords.data, 0, 2).astype(np.float32)
        return w

    @property
    def width(self):
        return 256

    @property
    def height(self):
        return 256

def image_to_QImage(image: ImageLike) -> QImage:
    # transform from 0..1 float to 0..255 int
    flat = image.clip(0, 1) * 255
    flat = flat.astype(np.uint32)

    # shift values into the correct position and create QImage from flat array
    flat = (255 << 24
        | flat[:,:,0] << 16
        | flat[:,:,1] << 8
        | flat[:,:,2]).flatten()
    img = QImage(flat,
        image.shape[1], image.shape[0],
        QImage.Format_ARGB32)

    return img

@njit(parallel=True)
def draw_color_wheel_lch(canvas: ImageLike):
    center = np.array((0.5, 0.5), dtype=np.float32)

    for y in prange(canvas.shape[0]):
        for x in prange(canvas.shape[1]):
            r_coords = np.array((y, x), dtype=np.float32) / np.array(canvas.shape[:2], dtype=np.float32)
            ry, rx = r_coords
            angle = np.arctan2(ry - 0.5, rx - 0.5)
            hue = np.degrees(angle) % 360
            chroma = np.linalg.norm(r_coords - center)
            canvas[y, x] = np.array((50, chroma * 2 * 132, hue)) # * (chroma < 0.5)

@njit(parallel=True)
def draw_color_wheel_hsv(canvas: ImageLike):
    center = np.array((0.5, 0.5), dtype=np.float32)

    for y in prange(canvas.shape[0]):
        for x in prange(canvas.shape[1]):
            r_coords = np.array((y, x), dtype=np.float32) / np.array(canvas.shape[:2], dtype=np.float32)
            ry, rx = r_coords
            sat = np.linalg.norm(r_coords - center)
            hue = np.arctan2(ry - 0.5, rx - 0.5) / (2 * pi)

            canvas[y, x] = np.array((hue, sat, 1), dtype=np.float32)

@njit(float32[:](float32[:]))
def xyY_to_XYZ(xyY: Union[npt.ArrayLike, Color]):
    x, y, Y = xyY
    if y == 0:
        return np.array((0, 0, 0), dtype=np.float32)
    X = (x * Y) / y
    Z = (1 - x - y) * Y / y

    return np.array((X, Y, Z), dtype=np.float32)


@njit(parallel=True)
def draw_color_wheel(canvas: ImageLike):
    for y in prange(canvas.shape[0]):
        for x in prange(canvas.shape[1]):
            vx = y / canvas.shape[0]
            vy = x / canvas.shape[1]
            canvas[y, x] = xyY_to_XYZ(np.array((vx, vy, 1), dtype=np.float32))