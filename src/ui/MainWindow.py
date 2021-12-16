################################################################################
# Entry point to the Graphical interface of the Neo Alchemist
################################################################################

from typing import Optional
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QDockWidget, QGridLayout, QGroupBox, QLabel, QSlider, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter
from numpy.lib.type_check import imag
from superqt import QRangeSlider
import numpy as np

import rawpy
from math import floor

class LabelledSlider(QWidget):
    def __init__(self, label: str, parent: Optional[QWidget] = None, f: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, f=f)

        self.label = QLabel(label, self)
        self.slider = QSlider(Qt.Horizontal, self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.setLayout(self.layout)

class PropertiesPane(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Properties", parent=parent)

        # self.exposure = LabelledSlider("Exposure", self)
        # self.white_balance = LabelledSlider("White Balance", self)
        # self.tint = LabelledSlider("Tint", self)

        self.shadows = LabelledSlider("Shadows", self)
        self.shadows.slider.setMinimum(-100)
        self.shadows.slider.setMaximum(100)
        self.shadows.slider.setValue(0)

        self.highlights = LabelledSlider("Highlights", self)
        self.highlights.slider.setMinimum(0)
        self.highlights.slider.setMaximum(200)
        self.highlights.slider.setValue(100)

        self.gamma = LabelledSlider("Gamma", self)
        self.gamma.slider.setMinimum(0)
        self.gamma.slider.setMaximum(1000)
        self.gamma.slider.setValue(100)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)
        # self._layout.addWidget(self.exposure)
        # self._layout.addWidget(self.white_balance)
        # self._layout.addWidget(self.tint)
        self._layout.addWidget(self.shadows)
        self._layout.addWidget(self.highlights)
        self._layout.addWidget(self.gamma)

        self.setLayout(self._layout)

class ImageRenderer(QOpenGLWidget):
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

####################################################
# Business logic
#
# TODO: Move to its own file
####################################################

def gamma(image, gamma):
    return image ** gamma

def linear_contrast(image, shadows: float, highlights: float):
    """
    Adjust contrast linearly to parameters
    """

    return image * highlights + shadows

def invert(image):
    # max_r = np.max(image[..., 0])
    # max_g = np.max(image[..., 1])
    # max_b = np.max(image[..., 2])

    return 1 - image

class Editor():
    def __init__(self, img_path) -> None:
        self._raw_data = rawpy.imread(img_path)

        self._uncorrected_linear = np.array(self._raw_data.postprocess(
            half_size=True,
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            no_auto_bright=True
            ), dtype=np.float32) / np.iinfo(np.uint16).max

        self._white_balance_sample_area = (0.45, 0.45, 0.01, 0.01)

        self.shadows = 0
        self.highlights = 1
        self.gamma = 1

        # self._linear_white_balanced = np.array(self._raw_data.postprocess(
        #     user_wb=self.white_balance,
        #     # output_color=rawpy.ColorSpace.ProPhoto,
        #     output_color=rawpy.ColorSpace.raw,
        #     output_bps=16
        #     ), dtype=np.float32) / np.iinfo(np.uint16).max

    def _fit_image(self, img, fit: tuple[int]) -> np.ndarray:
        w = img.shape[0]
        h = img.shape[1]
        _fit = (min(fit[0], w),
            min(fit[1], h))
        stride = max(1, floor(max(w/_fit[0], h/_fit[1])))
        return img[::stride, ::stride]

    def uncorrected_linear(self, fit: tuple[int]) -> np.ndarray:
        """
        The uncorrected image scaled to fit the resolution defined in `fit`
        """
        return self._fit_image(self._uncorrected_linear, fit)

    def white_balanced_linear(self, fit: tuple[int]) -> np.ndarray:
        """
        The white balanced image scaled to fit the resolution defined in `fit`
        """
        # return self._fit_image(self._linear_white_balanced, fit)
        fitted = self._fit_image(self._uncorrected_linear, fit)
        wb = self.white_balance

        fitted[:,:,0] *= wb[0]
        fitted[:,:,1] *= wb[1]
        fitted[:,:,2] *= wb[2]

        return fitted

    def inverted(self, fit: tuple[int]) -> np.ndarray:
        """
        The white balanced and iverted image scaled to fit the resolution
        defined in `fit`
        """
        return invert(self.white_balanced_linear(fit))

    def linear_contrast_adjusted(self, fit: tuple[int]) -> np.ndarray:
        return linear_contrast(self.inverted(fit), self.shadows, self.highlights)

    def gamma_adjusted(self, fit: tuple[int]) -> np.ndarray:
        return gamma(self.linear_contrast_adjusted(fit), self.gamma)

    def process(self):
        """
        Process the image with the set parameters
        """
        linear_white_balanced = np.array(self._raw_data.postprocess(
            user_wb=self.white_balance,
            # output_color=rawpy.ColorSpace.ProPhoto,
            output_color=rawpy.ColorSpace.raw,
            output_bps=16
            ), dtype=np.float32) / np.iinfo(np.uint16).max

        inverted = self.invert(linear_white_balanced)

        return inverted

    @property
    def white_balance_sample_area(self):
        """
        Area `(x, y, width, height)` over which the white balance is sampled
        """
        return self._white_balance_sample_area

    @white_balance_sample_area.setter
    def white_balance_sample_area(self, area: tuple[float]):
        self._white_balance_sample_area = area
        print(f"WB area: {self._white_balance_sample_area}")

        # self._linear_white_balanced = np.array(self._raw_data.postprocess(
        #     user_wb=self.white_balance,
        #     # output_color=rawpy.ColorSpace.ProPhoto,
        #     output_color=rawpy.ColorSpace.raw,
        #     output_bps=16
        #     ), dtype=np.float32) / np.iinfo(np.uint16).max

    @property
    def white_balance(self):
        """
        Calculate the white balance over the `white_balance_sample_area`
        """
        proxy = self.uncorrected_linear((256, 256))

        w = proxy.shape[0]
        h = proxy.shape[1]

        y1 = int(self._white_balance_sample_area[1] * h)
        y2 = int((self._white_balance_sample_area[1] + self._white_balance_sample_area[3]) * h)

        x1 = int(self._white_balance_sample_area[0] * w)
        x2 = int((self._white_balance_sample_area[0] + self._white_balance_sample_area[2]) * w)

        wb_patch = proxy[y1:y2, x1:x2]

        avg_r = np.average(wb_patch[..., 0])
        avg_g = np.average(wb_patch[..., 1])
        avg_b = np.average(wb_patch[..., 2])

        return [avg_g/avg_r, 1.0, avg_g/avg_b, 1.0]

####################################################
# End of Business logic
####################################################

class MainWindow(QWidget):
    def __init__(self, cl_args) -> None:
        super().__init__()

        # Data setup
        editor = Editor(cl_args.file)

        img = editor.gamma_adjusted((1024, 1024))

        # UI setup
        self.setWindowTitle("Neo Alchemist")

        # Properties panel
        props = PropertiesPane(self)

        viewer_group = QGroupBox("Viewer", self)
        viewer_layout = QGridLayout()

        viewer = ImageRenderer(img, self)
        vertical_wb_range = QRangeSlider()
        horizontal_wb_range = QRangeSlider(Qt.Horizontal)

        def update_viewer():
            viewer.image = editor.gamma_adjusted((1024, 1024))

            viewer.overlay = [QRect(
                editor.white_balance_sample_area[0] * viewer.width,
                editor.white_balance_sample_area[1] * viewer.height,
                editor.white_balance_sample_area[2] * viewer.width,
                editor.white_balance_sample_area[3] * viewer.height,
                )]

            viewer.update()

        def set_vertical_wb_area(values):
            current = editor.white_balance_sample_area
            editor.white_balance_sample_area = (
                current[0],
                (99 - values[1]) / 100,
                current[2],
                (values[1] - values[0]) / 100)

            update_viewer()

        set_vertical_wb_area(vertical_wb_range.value())
        vertical_wb_range.valueChanged.connect(set_vertical_wb_area)

        def set_horizontal_wb_area(values):
            current = editor.white_balance_sample_area
            editor.white_balance_sample_area = (
                values[0] / 100,
                current[1],
                (values[1] - values[0]) / 100,
                current[3])

            update_viewer()

        set_horizontal_wb_area(horizontal_wb_range.value())
        horizontal_wb_range.valueChanged.connect(set_horizontal_wb_area)

        def set_shadows(value):
            editor.shadows = value / 100
            update_viewer()

        props.shadows.slider.valueChanged.connect(set_shadows)

        def set_highlights(value):
            editor.highlights = value / 100
            update_viewer()

        props.highlights.slider.valueChanged.connect(set_highlights)

        def set_gamma(value):
            editor.gamma = value / 100
            update_viewer()

        props.gamma.slider.valueChanged.connect(set_gamma)

        viewer_layout.addWidget(vertical_wb_range, 0, 0)
        viewer_layout.addWidget(viewer, 0, 1)
        viewer_layout.addWidget(horizontal_wb_range, 1, 1)
        viewer_group.setLayout(viewer_layout)

        main_layout = QGridLayout()
        main_layout.setColumnMinimumWidth(0, 256)
        main_layout.addWidget(props, 0, 0)
        main_layout.addWidget(viewer_group, 0, 1)

        self.setLayout(main_layout)
