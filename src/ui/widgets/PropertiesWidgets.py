from typing import Optional
from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QCheckBox, \
    QFileDialog, \
    QGroupBox, \
    QLabel, \
    QPushButton, \
    QSpinBox, \
    QComboBox, \
    QVBoxLayout, \
    QWidget

import numpy.typing as npt

from .Atoms import ColorBalanceControl, ColorPicker, LabelledSlider

from ...processing.spells import Color

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

class ColorBalanceWidget(QGroupBox):
    """
    Widget to balance three primary colours
    """
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.color = ColorPicker(self)
        self.color.setMinimum(0.4)
        self.color.setMaximum(1.1)
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
        self.shadow_balance.setMinimum(-1)
        self.shadow_balance.setMaximum(1)
        self.shadow_balance.setValue(0)

        self._highlight_label = QLabel("Highlight Balance", self)
        self.highlight_balance = ColorPicker(self)
        self.highlight_balance.setMinimum(0)
        self.highlight_balance.setMaximum(2)
        self.highlight_balance.setValue(1)

        self.balance_control = ColorBalanceControl(self)
        self.balance_control.update()

        self._layout.addWidget(self._shadow_label)
        self._layout.addWidget(self.shadow_balance)
        self._layout.addWidget(self._highlight_label)
        self._layout.addWidget(self.highlight_balance)
        self._layout.addWidget(self.balance_control)

        self.setLayout(self._layout)

class CDLWidget(QGroupBox):
    """
    Widget exposing ASC CDL Primary grading controls
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._slope_label = QLabel("Slope", self)
        self.slope = ColorBalanceControl(self)
        self.slope.update()
        self._offset_label = QLabel("Offset", self)
        self.offset = ColorBalanceControl(self)
        self.offset.update()
        self._power_label = QLabel("Power", self)
        self.power = ColorBalanceControl(self)
        self.power.update()

        self._layout.addWidget(self._slope_label)
        self._layout.addWidget(self.slope)
        self._layout.addWidget(self._offset_label)
        self._layout.addWidget(self.offset)
        self._layout.addWidget(self._power_label)
        self._layout.addWidget(self.power)

        self.setLayout(self._layout)

    ################################################################
    # SLOPE ACCESSORS
    ################################################################

    def get_slope(self):
        return self.slope.value()

    def set_slope(self, value: npt.ArrayLike):
        return self.slope.set_value(value)

    @property
    def slope_changed(self):
        return self.slope.value_changed

    ################################################################
    # OFFSET ACCESSORS
    ################################################################

    def get_offset(self):
        return self.offset.value()

    def set_offset(self, value: npt.ArrayLike):
        return self.offset.set_value(value)

    @property
    def offset_changed(self):
        return self.slope.value_changed

    ################################################################
    # POWER ACCESSORS
    ################################################################

    def get_power(self):
        return self.power.value()

    def set_power(self, value: npt.ArrayLike):
        return self.power.set_value(value)

    @property
    def power_changed(self):
        return self.power.value_changed

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

class HueSatWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._hue = LabelledSlider("Hue", self)
        self._hue.setMinimum(-2)
        self._hue.setMaximum( 2)
        self._hue.setValue(0)
        self._layout.addWidget(self._hue)

        self._saturation = LabelledSlider("Saturation", self)
        self._saturation.setMinimum(0)
        self._saturation.setMaximum(2)
        self._saturation.setValue(1)
        self._layout.addWidget(self._saturation)

        self.setLayout(self._layout)

    def hue(self):
        return self._hue.value()

    def set_hue(self, value):
        return self._hue.setValue(value)

    @property
    def hue_changed(self):
        return self._hue.valueChanged

    def saturation(self):
        return self._saturation.value()

    def set_saturation(self, value):
        return self._saturation.setValue(value)

    @property
    def saturation_changed(self):
        return self._saturation.valueChanged

class SaturationWidget(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self._saturation = LabelledSlider("Saturation", self)
        self._saturation.setMinimum(0)
        self._saturation.setMaximum(2)
        self._saturation.setValue(1)
        self._layout.addWidget(self._saturation)

        self.setLayout(self._layout)

    def saturation(self):
        return self._saturation.value()

    def set_saturation(self, value):
        return self._saturation.setValue(value)

    @property
    def saturation_changed(self):
        return self._saturation.valueChanged

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
        self.factor.setMinimum(-1)
        self.factor.setMaximum(16)
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
    def __init__(self, title: str, colorspaces: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop)

        self.from_space_label = QLabel("From:")
        self._layout.addWidget(self.from_space_label)

        self.from_space = QComboBox(self)
        self.from_space.setInsertPolicy(QComboBox.NoInsert)
        self.from_space.addItems(colorspaces)
        self._layout.addWidget(self.from_space)

        self.to_space_label = QLabel("To:")
        self._layout.addWidget(self.to_space_label)

        self.to_space = QComboBox(self)
        self.to_space.setInsertPolicy(QComboBox.NoInsert)
        self.to_space.addItems(colorspaces)
        self._layout.addWidget(self.to_space)

        self.setLayout(self._layout)

    def get_from_space(self):
        return self.from_space.currentText()

    def set_from_space(self, value):
        self.from_space.setCurrentText(value)

    @property
    def from_space_changed(self):
        return self.from_space.currentTextChanged

    def get_to_space(self):
        return self.to_space.currentText()

    def set_to_space(self, value):
        self.to_space.setCurrentText(value)

    @property
    def to_space_changed(self):
        return self.to_space.currentTextChanged

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
