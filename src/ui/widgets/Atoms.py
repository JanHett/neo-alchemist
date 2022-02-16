from typing import Optional, Union
from PySide2.QtCore import QRect, Qt, Signal, QPoint
from PySide2.QtWidgets import \
    QGridLayout, \
    QLabel, \
    QComboBox, \
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

from numba import prange, njit

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

class ChromaHuePicker(QWidget):

    pos_changed = Signal(tuple)

    def __init__(self,
        parent: Optional[QWidget] = None,
        f: Qt.WindowFlags = Qt.WindowFlags()):
        super().__init__(parent, f)

        # size of the control
        self._width = 256
        self._height = 256

        self.setFixedSize(self.width, self.height)

        # lightness of the visualisation
        self._lightness = 0.5

        # colorspace used for visualisation
        self._display_colorspace = "Output - P3D65"
        # processor to transform XYZ representation to display space
        self._ocio_proc = global_ocio.get_processor("Utility - XYZ - D60",
            self._display_colorspace)

        # position of the marker
        self._pos = np.array((0.5, 0.5))

        # speed adjustment
        self._factor = 0.1

        # initial value for marker movement
        self._previous_pos = QPoint(0, 0)

        self._rerender()

    def paintEvent(self, event: QPaintEvent):
        w = self._display_wheel

        wheel = image_to_QImage(w)
        # TODO: make size more flexible
        self.setFixedSize(wheel.size())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # draw the image inside a circle
        clip_elipse = QPainterPath()
        clip_elipse.addEllipse(0, 0, wheel.width(), wheel.height())
        painter.setClipPath(clip_elipse)
        painter.drawImage(0, 0, wheel)

        # draw overlays
        painter.setPen(QColor(0, 255, 0))
        marker_x = int(self._pos[1] * wheel.width())
        marker_y = int(self._pos[0] * wheel.height())
        radius = 10
        painter.drawEllipse(
            marker_x - radius, marker_y - radius,
            2 * radius, 2 * radius)

    def set_lightness(self, L: float):
        self._lightness = L
        self._rerender()

    def set_display_colorspace(self, colorspace: str):
        self._display_colorspace = colorspace
        self._ocio_proc = global_ocio.get_processor("Utility - XYZ - D60",
            self._display_colorspace)
        self._rerender()

    def set_size(self, width, height):
        self._width = width
        self._height = height
        self._rerender()

    def set_speed_factor(self, factor: float):
        """
        Adjust the speed with which the control moves relative to the cursor
        """
        self._factor = factor

    def set_pos(self, pos: tuple[float, float]):
        """
        Set the position of the marker in [y, x] coordinates
        """
        self._pos = np.array(pos)
        self.pos_changed.emit(self._pos)
        self._rerender()

    def get_pos(self):
        """
        Get position of the marker in [y, x] coordinates
        """
        return self._pos

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    def _rerender(self):
        """
        Recalculate display wheel and update
        """
        self._display_wheel = self._generate_display_wheel()
        self.update()

    def _generate_display_wheel(self):
        w = np.zeros((self.height, self.width, 3), dtype=np.float32)
        draw_color_wheel_lch(w, self._lightness * 100)

        # format wheel for colorio and convert to XYZ
        w_moved = np.moveaxis(w, 2, 0)
        c_coords = ColorCoordinates(w_moved, CIELCH(whitepoint=D60))
        c_coords.convert(XYZ(1))

        # get wheel back to usual format and transform to display color space
        # TODO: without the copy the wheel looks odd - fix that
        w = np.moveaxis(c_coords.data, 0, 2).astype(np.float32).copy()
        self._ocio_proc.applyRGB(w)

        return w

    def mousePressEvent(self, event: QMouseEvent):
        self._previous_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        delta = event.pos() - self._previous_pos
        # convert delta to unit coordinates
        delta = np.array((delta.y(), delta.x())) / self._display_wheel.shape[:2]
        # print("delta:", delta)
        self._previous_pos = event.pos()
        self._pos = self._pos + delta * self._factor
        # print("self._pos:", self._pos)
        # print("self.xyz_color:", self.xyz_color)
        # print("self._p3_color:", self._p3_color)
        self.pos_changed.emit(self._pos)
        self.update()

class LChColorPicker(QWidget):
    color_changed = Signal(tuple)
    """
    Signal emitted when the color is changed. Handlers will be handed the
    selected colour in the specified output colour space
    """

    def __init__(self,
        parent: Optional[QWidget] = None,
        f: Qt.WindowFlags = Qt.WindowFlags()):
        """
        Construct an L*C*h* color picker
        """
        super().__init__(parent, f)

        #################################
        # Data setup, part 1
        #################################
        self._colorspaces = global_ocio.get_colorspaces()

        #################################
        # Presentation setup
        #################################
        self._layout = QVBoxLayout(self)

        self._chroma_hue = ChromaHuePicker(self)
        self._chroma_hue.pos_changed.connect(self._emit_color_changed)
        self._lightness = LabelledSlider("Lightness", self)
        self._lightness.valueChanged.connect(self._chroma_hue.set_lightness)
        self._lightness.setMinimum(0)
        self._lightness.setMaximum(1)
        self._lightness.setValue(0.5)
        self._lightness.valueChanged.connect(self._emit_color_changed)

        self._display_colorspace_label = QLabel("Display Colour Space", self)
        self._display_colorspace = QComboBox(self)
        self._display_colorspace.setInsertPolicy(QComboBox.NoInsert)
        self._display_colorspace.addItems(self._colorspaces)
        self._display_colorspace.currentTextChanged.connect(self._chroma_hue.set_display_colorspace)
        self._display_colorspace.setCurrentText("Role - color_picking")

        self._output_colorspace_label = QLabel("Output Colour Space", self)
        self._output_colorspace = QComboBox(self)
        self._output_colorspace.setInsertPolicy(QComboBox.NoInsert)
        self._output_colorspace.addItems(self._colorspaces)
        self._output_colorspace.setCurrentText("Role - default")
        self._output_colorspace.currentTextChanged.connect(self._emit_color_changed)

        #################################
        # Data setup, part 2
        #################################
        # the processor to convert from XYZ to the output color space
        self._output_ocio_proc = global_ocio.get_processor(
            "Utility - XYZ - D60",
            self.get_output_colorspace())
        # the processor to convert from the output colour space to XYZ (used for
        # setting the colour)
        self._input_ocio_proc = global_ocio.get_processor(
            self.get_output_colorspace(),
            "Utility - XYZ - D60")

        #################################
        # Presentation setup, cotinued
        #################################

        self._lch_value_label = QLabel("", self)
        self._output_value_label = QLabel("", self)
        self._update_value_labels(self.get_color())
        self.color_changed.connect(self._update_value_labels)

        # Laying out the colour picker itself...
        self._layout.addWidget(self._chroma_hue)
        self._layout.addWidget(self._lightness)

        # ...the labels for the current values
        self._layout.addWidget(self._lch_value_label)
        self._layout.addWidget(self._output_value_label)

        # ...and the display and output colour space slectors
        self._layout.addWidget(self._display_colorspace_label)
        self._layout.addWidget(self._display_colorspace)
        self._layout.addWidget(self._output_colorspace_label)
        self._layout.addWidget(self._output_colorspace)

        self.setLayout(self._layout)

        

    def get_display_colorspace(self):
        return self._display_colorspace.currentText()

    def set_display_colorspace(self, colorspace: str):
        self._display_colorspace.setCurrentText(colorspace)

    def get_output_colorspace(self):
        return self._output_colorspace.currentText()

    def set_output_colorspace(self, colorspace: str):
        self._output_colorspace.setCurrentText(colorspace)
        self._output_ocio_proc = global_ocio.get_processor(
            "Utility - XYZ - D60",
            self.get_output_colorspace())
        self._input_ocio_proc = global_ocio.get_processor(
            self.get_output_colorspace(),
            "Utility - XYZ - D60")

    def get_lch(self):
        ch_pos = self._chroma_hue.get_pos()
        lch = pos_to_lch(ch_pos, self._lightness.value() * 100)
        return lch

    def get_color(self):
        """
        Returns the currently selected colour in the currently specified output
        colour space
        """
        lch = self.get_lch()

        # format wheel for colorio and convert to XYZ
        # w_moved = np.moveaxis(w, 2, 0)
        c_coords = ColorCoordinates(lch, CIELCH(whitepoint=D60))
        c_coords.convert(XYZ(1))
        # TODO: XYZ values seem suspiciously low - fix that

        # get wheel back to usual format and transform to display color space
        # TODO: without the copy the wheel looks odd - fix that
        # c = np.moveaxis(c_coords.data, 0, 2).astype(np.float32).copy()
        c = c_coords.data.astype(np.float32).copy()
        self._output_ocio_proc.applyRGB(c)
        return c

    def set_color(self, color: Color) -> None:
        """
        Set the colour with reference to the currently selected output color
        space
        """
        # copy this to be safe from side effects on the callers side
        c_in = np.array(color, dtype=np.float32).copy()
        self._input_ocio_proc.applyRGB(c_in)
        c_coords = ColorCoordinates(c_in, XYZ(1))
        c_coords.convert(CIELCH(whitepoint=D60))
        c, h = c_coords.data[1:]
        
        self._chroma_hue.set_pos(chroma_hue_to_pos(c, h))
        self._lightness.setValue(c_coords.data[0] / 100)

    def _emit_color_changed(self, _):
        self.color_changed.emit(self.get_color())

    def _update_value_labels(self, color):
        lch_text = f"L*C*h*: {self.get_lch()}"
        output_text = f"Output: {color}"
        self._lch_value_label.setText(lch_text)
        self._output_value_label.setText(output_text)

# TODO: rename this to ChromaHuePicker or something of the sort
# class ColorBalanceControl(QOpenGLWidget):

#     value_changed = Signal(tuple)

#     def __init__(self,
#         # neutral_value = 1,
#         parent: Optional[QWidget] = None,
#         f: Qt.WindowFlags = Qt.WindowFlags()):
#         super().__init__(parent, f)
#         # TODO: make colour space customisable
#         self._ocio_proc = global_ocio.get_processor("Utility - XYZ - D60", "Output - P3D65")
#         # self._ocio_proc = global_ocio.get_processor("Role - scene_linear", "Output - P3D65")
#         self._xyz_wheel = self._generate_wheel()
#         self._p3_wheel = self._xyz_wheel.copy()
#         self._ocio_proc.applyRGB(self._p3_wheel)
#         self.setFixedSize(self.width, self.height)

#         self._previous_pos = QPoint(0, 0)
#         self._pos = np.array((0.5, 0.5))

#         self._factor = 0.1

#     def paintEvent(self, event: QPaintEvent):
#         w = self._p3_wheel
#         self.setFixedSize(self.width, self.height)
#         wheel = image_to_QImage(w)
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)

#         # draw the image inside a circle
#         clip_elipse = QPainterPath()
#         clip_elipse.addEllipse(0, 0, self.width, self.height)
#         painter.setClipPath(clip_elipse)
#         painter.drawImage(0, 0, wheel)
#         # draw overlays
#         painter.setPen(QColor(0, 255, 0))
#         marker_y, marker_x = (self._pos * self._xyz_wheel.shape[:2]).astype(np.int)
#         radius = 10
#         painter.drawEllipse(marker_x - radius, marker_y - radius, 2 * radius, 2 * radius)
#         # painter.drawRects(self._overlay)

#     def mousePressEvent(self, event: QMouseEvent):
#         self._previous_pos = event.pos()

#     def mouseMoveEvent(self, event: QMouseEvent):
#         delta = event.pos() - self._previous_pos
#         delta = np.array((delta.y(), delta.x())) / self._xyz_wheel.shape[:2]
#         # print("delta:", delta)
#         self._previous_pos = event.pos()
#         self._pos = self._pos + delta * self._factor
#         # print("self._pos:", self._pos)
#         # print("self.xyz_color:", self.xyz_color)
#         # print("self._p3_color:", self._p3_color)
#         self.value_changed.emit(self.xyz_color)
#         self.update()

#     @property
#     def xyz_color(self):
#         y, x = (self._pos * self._xyz_wheel.shape[:2]).astype(np.int)
#         return self._xyz_wheel[y, x]

#     @property
#     def _p3_color(self):
#         c = self.xyz_color.copy()
#         self._ocio_proc.applyRGB(c)
#         return c

#     @property
#     def position(self):
#         """
#         Position of the picker in unit coordinates
#         """
#         return self._pos

#     def get_value(self):
#         return self.xyz_color

#     def set_value(self, value):
#         # TODO
#         pass

#     def _generate_wheel(self):
#         w = np.zeros((self.height, self.width, 3), dtype=np.float32)
#         draw_color_wheel_lch(w)
#         # w = make_wheel()
#         w_moved = np.moveaxis(w, 2, 0)
#         c_coords = ColorCoordinates(w_moved, CIELCH(whitepoint=D60))
#         c_coords.convert(XYZ(1))
#         w = np.moveaxis(c_coords.data, 0, 2).astype(np.float32)
#         return w

#     @property
#     def width(self):
#         return 256

#     @property
#     def height(self):
#         return 256

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
def draw_color_wheel_lch(canvas: ImageLike, lightness: float = 50.):
    for y in prange(canvas.shape[0]):
        for x in prange(canvas.shape[1]):
            r_coords = np.array((y, x)) / np.array(canvas.shape[:2])
            canvas[y, x] = pos_to_lch(r_coords, lightness)

@njit
def pos_to_lch(pos: npt.ArrayLike, L: float) -> npt.ArrayLike:
    """
    Calculate Chroma and Hue based on position in unit coordinates
    """
    # position relative to 0.5, 0.5 center
    rpos = pos - np.array((0.5, 0.5))
    y, x = rpos
    angle = np.arctan2(y, x)
    hue = np.degrees(angle) % 360
    chroma = np.linalg.norm(rpos)
    return np.array((L, chroma * 2 * 132, hue))

def chroma_hue_to_pos(chroma: float, hue: float) -> npt.ArrayLike:
    """
    Calculate position on chroma/hue circle in cartesian unit coordinates from
    chroma and hue
    """
    _h = np.radians(hue)
    _c = chroma / (2 * 132)
    a, b = _c * np.cos(_h), _c * np.sin(_h)
    return np.array((a, b)) + 0.5

class BetterColorPicker(QWidget):
    """
    A general purpose color picker
    """

    LCh_changed = Signal(tuple)
    XYZ_changed = Signal(tuple)
    Lab_changed = Signal(tuple)

    def __init__(self, colorspaces: list[str], parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent, f)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.wheel = ColorBalanceControl(self)
        self.lightness = LabelledSlider("L*", self)
        self.colorspace = QComboBox(self)
        self.from_space.setInsertPolicy(QComboBox.NoInsert)
        self.from_space.addItems(colorspaces)
        self._layout.addWidget(self.from_space)

        self._layout.addWidget(self.wheel)

        self.setLayout(self._layout)

        self._ocio_proc = global_ocio.get_processor(self.get_colorspace(), "Utility - XYZ - D60")

    def setValue(self, value: Union[Color, float]) -> None:
        """
        Set value in `colorspace`
        """
        try:
            color = np.array(value, dtype=np.float32)
        except TypeError:
            color = np.array((value, value, value), dtype=np.float32)

        self._ocio_proc.applyRGB(color)

    def value(self) -> Color:
        return pos_to_lch(self.wheel.position, self.lightness.value)

    def set_LCh(self, value: Union[Color, float]) -> None:
        """
        Set value in L*C*h*
        """
        pass


    def _emit_color_changed(self, _):
        self.LCh_changed.emit(self.value())

    def get_colorspace(self):
        return self.colorspace.currentText()

    def set_colorspace(self, value):
        self.colorspace.setCurrentText(value)

    @property
    def colorspace_changed(self):
        return self.colorspace.currentTextChanged

    #############################################
    # COMPATIBILITY FUNCTIONS - REMOVE ASAP
    #############################################

    def setMinimum(self, value: Union[Color, float]) -> None:
        pass

    def setMaximum(self, value: Union[Color, float]) -> None:
        pass

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

        # self.wheel = ColorBalanceControl(self)

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