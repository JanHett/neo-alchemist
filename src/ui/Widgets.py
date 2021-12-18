from typing import Optional
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QDockWidget, QGridLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter
from superqt import QRangeSlider, QDoubleSlider

import numpy as np

class LabelledSlider(QWidget):
    def __init__(self, label: str, parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, f=f)

        self.label = QLabel(label, self)
        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.spinbox.setDecimals(4)
        self.slider = QDoubleSlider(Qt.Horizontal, self)

        def update_spinbox():
            self.spinbox.setValue(self.slider.value())

        def update_slider():
            self.slider.setValue(self.spinbox.value())

        self.slider.valueChanged.connect(update_spinbox)
        self.spinbox.editingFinished.connect(update_slider)

        self._layout = QGridLayout(self)
        self._layout.setColumnStretch(0, 2)
        self._layout.setColumnStretch(1, 1)
        self._layout.addWidget(self.label, 0, 0)
        self._layout.addWidget(self.spinbox, 0, 1)
        self._layout.addWidget(self.slider, 1, 0, 1, 2)

        self.setLayout(self._layout)

    def setValue(self, v: float) -> None:
        self.slider.setValue(v)
        self.spinbox.setValue(v)

    def setMaximum(self, v: float) -> None:
        self.slider.setMaximum(v)
        self.spinbox.setMaximum(v)

    def setMinimum(self, v: float) -> None:
        self.slider.setMinimum(v)
        self.spinbox.setMinimum(v)

class ColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self.shadows_group)
        self._layout.setAlignment(Qt.AlignTop)

        minimum = 0.8
        maximum = 1.2

        self.red = LabelledSlider("Highlights: Red", self.highlights_group)
        self.red.setMinimum(minimum)
        self.red.setMaximum(maximum)
        self.red.setValue(1)
        self.green = LabelledSlider("Highlights: Green", self.highlights_group)
        self.green.setMinimum(minimum)
        self.green.setMaximum(maximum)
        self.green.setValue(1)
        self.blue = LabelledSlider("Highlights: Blue", self.highlights_group)
        self.blue.setMinimum(minimum)
        self.blue.setMaximum(maximum)
        self.blue.setValue(1)

        self._layout.addWidget(self.highlights)
        self._layout.addWidget(self.red)
        self._layout.addWidget(self.green)
        self._layout.addWidget(self.blue)

        self.setLayout(self.highlights_layout)

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
        self.image = image
        self.setFixedSize(self.image.shape[1], self.image.shape[0])

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