from typing import Literal, Optional
from PySide2.QtCore import QRect, Qt, Signal
from PySide2.QtWidgets import QCheckBox, QDockWidget, QFileDialog, QGridLayout, QGroupBox, QLabel, QOpenGLWidget, QPushButton, QSlider, QDoubleSpinBox, QSpinBox, QVBoxLayout, QWidget
# from PySide2.QtOpenGLWidgets import QOpenGLWidget
from PySide2.QtGui import QColor, QImage, QPaintEvent, QPainter
from superqt import QRangeSlider, QDoubleSlider

import numpy as np
import numpy.typing as npt

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

class FileIOWidget(QGroupBox):
    """
    Widget providing a file picker
    """

    name_filter = "Any File (*)"
    filename = ""

    filename_changed = Signal(str)

    def __init__(self, title: str,
        parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

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

class FileInputWidget(FileIOWidget):
    def _handle_file_picker_click(self):
        self.filename = QFileDialog.getOpenFileName(self,
            "Open Image", "/", self.name_filter)[0]

        print(f"filename: '{self.filename}'")

class RawFileInputWidget(FileInputWidget):
    pass

class FileOutputWidget(FileIOWidget):
    def _handle_file_picker_click(self):
        self.filename = QFileDialog.getSaveFileName(self,
            "Save File", "/", self.name_filter)[0]

class ColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.activated = QCheckBox("Activated", self)
        self._layout.addWidget(self.activated)

        minimum = 0.8
        maximum = 1.2

        self.red = LabelledSlider("Red", self)
        self.red.setMinimum(minimum)
        self.red.setMaximum(maximum)
        self.red.setValue(1)
        self.green = LabelledSlider("Green", self)
        self.green.setMinimum(minimum)
        self.green.setMaximum(maximum)
        self.green.setValue(1)
        self.blue = LabelledSlider("Blue", self)
        self.blue.setMinimum(minimum)
        self.blue.setMaximum(maximum)
        self.blue.setValue(1)

        self._layout.addWidget(self.red)
        self._layout.addWidget(self.green)
        self._layout.addWidget(self.blue)

        self.setLayout(self._layout)

    def setMinimum(self, value: float):
        """
        Set minimum for all channel sliders
        """
        self.red.setMinimum(value)
        self.green.setMinimum(value)
        self.blue.setMinimum(value)

    def setMaximum(self, value: float):
        """
        Set maximum for all channel sliders
        """
        self.red.setMaximum(value)
        self.green.setMaximum(value)
        self.blue.setMaximum(value)

class TwoPointColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.activated = QCheckBox("Activated", self)
        self._layout.addWidget(self.activated)

        minimum = 0.8
        maximum = 1.2

        self.red = LabelledSlider("Red", self)
        self.red.setMinimum(minimum)
        self.red.setMaximum(maximum)
        self.red.setValue(1)
        self.green = LabelledSlider("Green", self)
        self.green.setMinimum(minimum)
        self.green.setMaximum(maximum)
        self.green.setValue(1)
        self.blue = LabelledSlider("Blue", self)
        self.blue.setMinimum(minimum)
        self.blue.setMaximum(maximum)
        self.blue.setValue(1)

        self._layout.addWidget(self.red)
        self._layout.addWidget(self.green)
        self._layout.addWidget(self.blue)

        self.setLayout(self._layout)

    def setMinimum(self, value: float):
        """
        Set minimum for all channel sliders
        """
        self.red.setMinimum(value)
        self.green.setMinimum(value)
        self.blue.setMinimum(value)

    def setMaximum(self, value: float):
        """
        Set maximum for all channel sliders
        """
        self.red.setMaximum(value)
        self.green.setMaximum(value)
        self.blue.setMaximum(value)

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

class ViewerOutputWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.width_input = QSpinBox(self)
        self.width_input.setMinimum(1)
        self.width_input.setMaximum(16777216)
        self._layout.addWidget(self.width_input)

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

    def set_image(self, image: npt.ArrayLike):
        self.image = image
        self.setFixedSize(self.image.shape[1], self.image.shape[0])