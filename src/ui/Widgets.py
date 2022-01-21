from typing import Literal, Optional, Union
from PySide2.QtCore import QRect, Qt, Signal
from PySide2.QtWidgets import QCheckBox, QDockWidget, QFileDialog, QGridLayout, QGroupBox, QLabel, QOpenGLWidget, QPushButton, QSlider, QDoubleSpinBox, QSpinBox, QVBoxLayout, QWidget
# from PySide2.QtOpenGLWidgets import QOpenGLWidget
from PySide2.QtGui import QColor, QImage, QPaintEvent, QPainter
from superqt import QRangeSlider, QDoubleSlider

import numpy as np
import numpy.typing as npt

from ..processing.spells import ImageLike, Color

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

class SolidWidget(QGroupBox):

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.color = ColorPicker(self)
        self.color.setMinimum(0)
        self.color.setMaximum(1)
        self.color.setValue(0.5)

        self._layout.addWidget(self.color)

        self.setLayout(self._layout)

    def get_color(self) -> Color:
        return self.color.value()

    def set_color(self, color: Color):
        self.color.setValue(color)

    @property
    def color_changed(self) -> Signal:
        return self.color.color_changed

class FileIOWidget(QGroupBox):
    """
    Widget providing a file picker
    """

    name_filter = "Any File (*)"

    filename_changed = Signal(str)

    def __init__(self, title: str,
        parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._filename = ""

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.filename_label = QLabel(self.filename)
        self.file_picker_button = QPushButton("Choose...", self)
        self.file_picker_button.clicked.connect(self.__handle_file_picker_click)
        self.filename_changed.connect(self.filename_label.setText)

        self._layout.addWidget(self.filename_label)
        self._layout.addWidget(self.file_picker_button)

        self.setLayout(self._layout)

    def __handle_file_picker_click(self):
        self._handle_file_picker_click()
        self.filename_changed.emit(self.filename)

    def get_filename(self):
        return self.filename

    @property
    def filename(self):
        return self._filename

    def set_filename(self, value):
        self._filename = value
        self.filename_changed.emit(self.filename)

    @filename.setter
    def filename(self, value):
        self.set_filename(value)


class FileInputWidget(FileIOWidget):
    def _handle_file_picker_click(self):
        self.filename = QFileDialog.getOpenFileName(self,
            "Open Image", "/", self.name_filter)[0]

        print(f"filename: '{self.filename}'")

class RawFileInputWidget(FileInputWidget):
    pass

class FileOutputWidget(FileIOWidget):
    def __init__(self, title: str,
        parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self.process_button = QPushButton("Process", self)

        self._layout.addWidget(self.process_button)

    def _handle_file_picker_click(self):
        self.filename = QFileDialog.getSaveFileName(self,
            "Save File", "/", self.name_filter)[0]

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

class ColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.color = ColorPicker(self)
        self.color.setMinimum(0.8)
        self.color.setMaximum(1.2)
        self.color.setValue(1)
        self._layout.addWidget(self.color)

        self.setLayout(self._layout)

class TwoPointColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._shadow_label = QLabel("Shadow Balance", self)
        self.shadow_balance = ColorPicker(self)
        self.shadow_balance.setMinimum(0)
        self.shadow_balance.setMaximum(1)
        self.shadow_balance.setValue(0.2)

        self._highlight_label = QLabel("Highlight Balance", self)
        self.highlight_balance = ColorPicker(self)
        self.highlight_balance.setMinimum(0)
        self.highlight_balance.setMaximum(1)
        self.highlight_balance.setValue(0.8)

        self._layout.addWidget(self._shadow_label)
        self._layout.addWidget(self.shadow_balance)
        self._layout.addWidget(self._highlight_label)
        self._layout.addWidget(self.highlight_balance)

        self.setLayout(self._layout)

class CropWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.crop_tool_button = QPushButton("Crop", self)
        # TODO: connect clicked signal (probably from outside)

        self.setLayout(self._layout)

class InvertWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.activated = QCheckBox("Activated", self)
        self._layout.addWidget(self.activated)

        self.setLayout(self._layout)

class GammaWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._gamma = LabelledSlider("Gamma", self)
        self._gamma.setMinimum(0)
        self._gamma.setMaximum(8)
        self._gamma.setValue(1)
        self._layout.addWidget(self._gamma)

        self.setLayout(self._layout)

    def gamma(self):
        return self._gamma.value()

    def set_gamma(self, value):
        return self._gamma.setValue(value)

    @property
    def gamma_changed(self):
        return self._gamma.valueChanged

class ContrastWidget(QGroupBox):
    parameters_changed = Signal(float, float)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._contrast = LabelledSlider("Contrast", self)
        self._contrast.setMinimum(0)
        self._contrast.setMaximum(2)
        self._contrast.setValue(1)
        self._layout.addWidget(self._contrast)

        self._lift = LabelledSlider("Lift", self)
        self._lift.setMinimum(-0.5)
        self._lift.setMaximum(1)
        self._lift.setValue(0)
        self._layout.addWidget(self._lift)

        self.contrast_changed.connect(self._emit_parameters_changed)
        self.lift_changed.connect(self._emit_parameters_changed)

        self.setLayout(self._layout)

    def contrast(self):
        return self._contrast.value()

    def set_contrast(self, value):
        return self._contrast.setValue(value)

    @property
    def contrast_changed(self):
        return self._contrast.valueChanged

    def lift(self):
        return self._lift.value()

    def set_lift(self, value):
        return self._lift.setValue(value)

    @property
    def lift_changed(self):
        return self._lift.valueChanged

    def _emit_parameters_changed(self):
        self.parameters_changed.emit

class EstimateColorBalanceWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

class PerChannelAverageWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

class AddWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.summand = LabelledSlider("Summand")
        self.summand.setMinimum(-8)
        self.summand.setMaximum( 8)
        self.summand.setValue(1)
        self._layout.addWidget(self.summand)

        self.setLayout(self._layout)

    def get_summand(self):
        return self.summand.value()

    def set_summand(self, summand):
        self.summand.setValue(summand)

    @property
    def summand_changed(self):
        return self.summand.valueChanged

class MultiplyWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.factor = LabelledSlider("Factor")
        self.factor.setMinimum(-8)
        self.factor.setMaximum( 8)
        self.factor.setValue(1)
        self._layout.addWidget(self.factor)

        self.setLayout(self._layout)

    def get_factor(self):
        return self.factor.value()

    def set_factor(self, factor):
        self.factor.setValue(factor)

    @property
    def factor_changed(self):
        return self.factor.valueChanged

class EqualsWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.comparison = LabelledSlider("Comparison")
        self.comparison.setMinimum(-8)
        self.comparison.setMaximum( 8)
        self.comparison.setValue(1)
        self._layout.addWidget(self.comparison)

        self.setLayout(self._layout)

    def get_comparison(self):
        return self.comparison.value()

    def set_comparison(self, comparison):
        self.comparison.setValue(comparison)

    @property
    def comparison_changed(self):
        return self.comparison.valueChanged

class LessThanWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.threshold = LabelledSlider("Threshold")
        self.threshold.setMinimum(-8)
        self.threshold.setMaximum( 8)
        self.threshold.setValue(1)
        self._layout.addWidget(self.threshold)

        self.setLayout(self._layout)

    def get_threshold(self):
        return self.threshold.value()

    def set_threshold(self, threshold):
        self.threshold.setValue(threshold)

    @property
    def threshold_changed(self):
        return self.threshold.valueChanged

class GreaterThanWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.threshold = LabelledSlider("Threshold")
        self.threshold.setMinimum(-8)
        self.threshold.setMaximum( 8)
        self.threshold.setValue(1)
        self._layout.addWidget(self.threshold)

        self.setLayout(self._layout)

    def get_threshold(self):
        return self.threshold.value()

    def set_threshold(self, threshold):
        self.threshold.setValue(threshold)

    @property
    def threshold_changed(self):
        return self.threshold.valueChanged

class OrWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

class AndWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

class ColorSpaceTransformWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

class ViewerOutputWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.width_label = QLabel("Width")
        self._layout.addWidget(self.width_label)
        self.width_input = QSpinBox(self)
        self.width_input.setMinimum(1)
        self.width_input.setMaximum(16777216)
        self._layout.addWidget(self.width_input)

        self.height_label = QLabel("Height")
        self._layout.addWidget(self.height_label)
        self.height_input = QSpinBox(self)
        self.height_input.setMinimum(1)
        self.height_input.setMaximum(16777216)
        self._layout.addWidget(self.height_input)

        self.setLayout(self._layout)

class PipelineWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None,
        f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, f=f)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)
        self.setLayout(self._layout)

    def push_step_widget(self, widget: QWidget):
        self._layout.addWidget(widget)

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
        # transform from 0..1 float to 0..255 int
        flat = self.image.clip(0, 1) * 255
        flat = flat.astype(np.uint32)

        # shift values into the correct position and create QImage from flat array
        flat = (255 << 24
            | flat[:,:,0] << 16
            | flat[:,:,1] << 8
            | flat[:,:,2]).flatten()
        img = QImage(flat,
            self.image.shape[1], self.image.shape[0],
            QImage.Format_ARGB32)

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