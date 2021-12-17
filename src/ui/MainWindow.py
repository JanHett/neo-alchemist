################################################################################
# Entry point to the Graphical interface of the Neo Alchemist
################################################################################

from typing import Optional
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QDockWidget, QGridLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter
from numpy.lib.type_check import imag
from superqt import QRangeSlider, QDoubleSlider
import numpy as np

import rawpy
from math import floor

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

        self.slider.sliderReleased.connect(update_spinbox)
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

class PropertiesPane(QGroupBox):
    def __init__(self, histogram, parent: Optional[QWidget] = None) -> None:
        super().__init__("Properties", parent=parent)

        # self.exposure = LabelledSlider("Exposure", self)
        # self.white_balance = LabelledSlider("White Balance", self)
        # self.tint = LabelledSlider("Tint", self)

        self.histogram = ImageRenderer(histogram)

        ### GAMMA ###

        self.gamma = LabelledSlider("Gamma", self)
        self.gamma.setMinimum(0)
        self.gamma.setMaximum(10)
        self.gamma.setValue(1)

        ### SHADOWS ###

        self.shadows_group = QGroupBox("Shadows", self)
        self.shadows_layout = QVBoxLayout(self.shadows_group)
        self.shadows_layout.setAlignment(Qt.AlignTop)

        self.shadows = LabelledSlider("Shadows: Main", self.shadows_group)
        self.shadows.setMinimum(-1)
        self.shadows.setMaximum(1)
        self.shadows.setValue(0)

        self.shadow_red = LabelledSlider("Shadows: Red", self.shadows_group)
        self.shadow_red.setMinimum(-0.3)
        self.shadow_red.setMaximum(0.3)
        self.shadow_red.setValue(0)
        self.shadow_green = LabelledSlider("Shadows: Green", self.shadows_group)
        self.shadow_green.setMinimum(-0.3)
        self.shadow_green.setMaximum(0.3)
        self.shadow_green.setValue(0)
        self.shadow_blue = LabelledSlider("Shadows: Blue", self.shadows_group)
        self.shadow_blue.setMinimum(-0.3)
        self.shadow_blue.setMaximum(0.3)
        self.shadow_blue.setValue(0)

        self.shadows_layout.addWidget(self.shadows)
        self.shadows_layout.addWidget(self.shadow_red)
        self.shadows_layout.addWidget(self.shadow_green)
        self.shadows_layout.addWidget(self.shadow_blue)

        self.shadows_group.setLayout(self.shadows_layout)

        ### HIGHLIGHTS ###

        self.highlights_group = QGroupBox("Highlights", self)
        self.highlights_layout = QVBoxLayout(self.shadows_group)
        self.highlights_layout.setAlignment(Qt.AlignTop)

        self.highlights = LabelledSlider("Highlights: Main", self.highlights_group)
        self.highlights.setMinimum(0)
        self.highlights.setMaximum(2)
        self.highlights.setValue(1)

        self.high_red = LabelledSlider("Highlights: Red", self.highlights_group)
        self.high_red.setMinimum(0.9)
        self.high_red.setMaximum(1.1)
        self.high_red.setValue(1)
        self.high_green = LabelledSlider("Highlights: Green", self.highlights_group)
        self.high_green.setMinimum(0.9)
        self.high_green.setMaximum(1.1)
        self.high_green.setValue(1)
        self.high_blue = LabelledSlider("Highlights: Blue", self.highlights_group)
        self.high_blue.setMinimum(0.9)
        self.high_blue.setMaximum(1.1)
        self.high_blue.setValue(1)

        self.highlights_layout.addWidget(self.highlights)
        self.highlights_layout.addWidget(self.high_red)
        self.highlights_layout.addWidget(self.high_green)
        self.highlights_layout.addWidget(self.high_blue)

        self.highlights_group.setLayout(self.highlights_layout)

        ### GENERAL SETUP ###

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._layout.addWidget(self.histogram)

        self._layout.addWidget(self.gamma)

        self._layout.addWidget(self.shadows_group)
        self._layout.addWidget(self.highlights_group)

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
    Adjust the balance of the highlights
    """
    shadows = np.array((shadow_red, shadow_green, shadow_blue))
    highs = np.array((high_red, high_green, high_blue))

    slope = highs - shadows
    return image * slope + shadows

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

        self.shadow_red = 0
        self.shadow_green = 0
        self.shadow_blue = 0

        self.high_red = 1
        self.high_green = 1
        self.high_blue = 1

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

    def highlow_balance_adjusted(self, fit: tuple[int]) -> np.ndarray:
        return highlow_balance(self.linear_contrast_adjusted(fit),
            self.shadow_red,
            self.shadow_green,
            self.shadow_blue,
            self.high_red,
            self.high_green,
            self.high_blue,
            )

    def gamma_adjusted(self, fit: tuple[int]) -> np.ndarray:
        return gamma(self.highlow_balance_adjusted(fit), self.gamma)

    def process(self):
        """
        Process the image with the set parameters
        """
        out = np.array(self._raw_data.postprocess(
            user_wb=self.white_balance,
            # output_color=rawpy.ColorSpace.ProPhoto,
            output_color=rawpy.ColorSpace.raw,
            output_bps=16
            ), dtype=np.float32) / np.iinfo(np.uint16).max

        out = invert(out)
        out = linear_contrast(out, self.shadows, self.highlights)
        out = gamma(out, self.gamma)

        return out

    @property
    def white_balance_sample_area(self):
        """
        Area `(x, y, width, height)` over which the white balance is sampled
        """
        return self._white_balance_sample_area

    @white_balance_sample_area.setter
    def white_balance_sample_area(self, area: tuple[float]):
        self._white_balance_sample_area = area

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

        def get_image(fit: tuple[int]) -> np.ndarray:
            return editor.gamma_adjusted(fit)

        img = get_image((1024, 1024))

        # UI setup
        self.setWindowTitle("Neo Alchemist")

        viewer_group = QGroupBox("Viewer", self)
        viewer_layout = QGridLayout()

        viewer = ImageRenderer(img, self)
        vertical_wb_range = QRangeSlider()
        horizontal_wb_range = QRangeSlider(Qt.Horizontal)

        def compute_histogram(buckets = 256, height = 128):
            img = get_image((256, 256))

            r_hist, _ = np.histogram(img[:,:,0], bins=buckets, range=(0, 1))
            g_hist, _ = np.histogram(img[:,:,1], bins=buckets, range=(0, 1))
            b_hist, _ = np.histogram(img[:,:,2], bins=buckets, range=(0, 1))

            hist_img = np.zeros((height, 256, 3), dtype=np.float32)

            max_v = np.max(r_hist)
            max_v = max(np.max(g_hist), max_v)
            max_v = max(np.max(b_hist), max_v)

            scale = height / max_v
            for x in range(buckets):
                hist_img[height - int(scale * r_hist[x]):, x, 0] = 1
                hist_img[height - int(scale * g_hist[x]):, x, 1] = 1
                hist_img[height - int(scale * b_hist[x]):, x, 2] = 1

            return hist_img

        # Properties panel
        props = PropertiesPane(compute_histogram(), self)

        def update_viewer():
            viewer.image = get_image((1024, 1024))
            props.histogram.image = compute_histogram()

            viewer.overlay = [QRect(
                editor.white_balance_sample_area[0] * viewer.width,
                editor.white_balance_sample_area[1] * viewer.height,
                editor.white_balance_sample_area[2] * viewer.width,
                editor.white_balance_sample_area[3] * viewer.height,
                )]

            viewer.update()
            props.histogram.update()

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
            editor.shadows = value
            update_viewer()

        props.shadows.slider.valueChanged.connect(set_shadows)

        def set_shadow_red(value):
            editor.shadow_red = value
            update_viewer()

        props.shadow_red.slider.valueChanged.connect(set_shadow_red)

        def set_shadow_green(value):
            editor.shadow_green = value
            update_viewer()

        props.shadow_green.slider.valueChanged.connect(set_shadow_green)

        def set_shadow_blue(value):
            editor.shadow_blue = value
            update_viewer()

        props.shadow_blue.slider.valueChanged.connect(set_shadow_blue)

        def set_highlights(value):
            editor.highlights = value
            update_viewer()

        props.highlights.slider.valueChanged.connect(set_highlights)

        def set_high_red(value):
            editor.high_red = value
            update_viewer()

        props.high_red.slider.valueChanged.connect(set_high_red)

        def set_high_green(value):
            editor.high_green = value
            update_viewer()

        props.high_green.slider.valueChanged.connect(set_high_green)

        def set_high_blue(value):
            editor.high_blue = value
            update_viewer()

        props.high_blue.slider.valueChanged.connect(set_high_blue)

        def set_gamma(value):
            editor.gamma = value
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
