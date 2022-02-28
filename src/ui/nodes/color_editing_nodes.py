from typing import Optional

from NodeGraphQt import Port
import numpy.typing as npt
import numpy as np

from ...util.data_storage_utils import np_to_list

from ...processing.spells import (
    cdl,
    gamma,
    hue_sat,
    invert,
    linear_contrast,
    saturation,
    two_point_color_balance,
    white_balance)

from ..widgets.PropertiesWidgets import (
    CDLWidget,
    ColorBalanceWidget,
    ContrastWidget,
    EstimateColorBalanceWidget,
    GammaWidget,
    HueSatWidget,
    InvertWidget,
    PerChannelAverageWidget,
    SaturationWidget,
    TwoPointColorBalanceWidget
    )

from .node_basics import NeoAlchemistNode, ImageCache, ROI

class GreyBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Grey Balance"
    def __init__(self):
        super().__init__()

        self.define_input("Image")
        image_out = self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = ColorBalanceWidget(self.NODE_NAME)

        self.reactive_property("coefficients", (1, 1, 1),
            self._properties_widget.color.value,
            self._properties_widget.color.setValue,
            self._properties_widget.color.color_changed)

    @property
    def grey_balance(self):
        return self.get_property("coefficients")

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return white_balance(in_img, self.grey_balance)

class TwoPointColorBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Two-Point Color Balance"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_input("Highlight Sample")
        self.define_input("Shadow Sample")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = TwoPointColorBalanceWidget(self.NODE_NAME)

        self.reactive_property("shadow_balance", 0.0,
            self._properties_widget.shadow_balance.value,
            self._properties_widget.shadow_balance.setValue,
            self._properties_widget.shadow_balance.color_changed)

        self.reactive_property("highlight_balance", 1.0,
            self._properties_widget.highlight_balance.value,
            self._properties_widget.highlight_balance.setValue,
            self._properties_widget.highlight_balance.color_changed)

    def _handle_request_image_data(self, roi: ROI):
        return two_point_color_balance(self.in_value("Image").get(roi),
            self.shadow_balance, self.highlight_balance)

    @property
    def shadow_balance(self):
        base = np.array((0, 0, 0), dtype=np.float32)
        if self.is_input_connected("Shadow Sample"):
            base = self.in_value("Shadow Sample")

        offset = self.get_property("shadow_balance")

        return base + offset

    @property
    def highlight_balance(self):
        base = np.array((0, 0, 0), dtype=np.float32)
        if self.is_input_connected("Highlight Sample"):
            base = self.in_value("Highlight Sample")

        offset = self.get_property("highlight_balance")

        return base + offset

    def on_input_connected(self, in_port: Port, out_port: Port):
        if (in_port.name() in ("Highlight Sample", "Shadow Sample")):
            self.set_property("shadow_balance", 0)
            self.set_property("highlight_balance", 0)
        return super().on_input_connected(in_port, out_port)

    def on_input_disconnected(self, in_port: Port, out_port: Port):
        # TODO: reset shadow and hoghlight balance after disconnection
        # self.set_property("shadow_balance", old_shadow_balance)
        # self.set_property("highlight_balance", old_highlight_balance)
        return super().on_input_disconnected(in_port, out_port)

class CDLNode(NeoAlchemistNode):
    NODE_NAME = "CDL (ASC Colour Decision List)"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = CDLWidget(self.NODE_NAME)

        # watch(self._model._custom_prop, cmp=lambda a, b: not np.array_equal(a, b))

        self.reactive_property("slope", (1, 1, 1),
            self._properties_widget.get_slope,
            self._properties_widget.set_slope,
            self._properties_widget.slope_changed,
            np_to_list)

        self.reactive_property("offset", (1, 1, 1),
            self._properties_widget.get_offset,
            self._properties_widget.set_offset,
            self._properties_widget.offset_changed,
            np_to_list)

        self.reactive_property("power", (1, 1, 1),
            self._properties_widget.get_power,
            self._properties_widget.set_power,
            self._properties_widget.power_changed,
            np_to_list)

    def _handle_request_image_data(self, roi: ROI):
        return cdl(self.in_value("Image").get(roi),
            self.slope, self.offset, self.power)

    @property
    def slope(self):
        return self._properties_widget.get_slope()
        # return self.get_property("slope")

    @property
    def offset(self):
        return self._properties_widget.get_offset() - 1
        # return self.get_property("offset") - 1

    @property
    def power(self):
        return self._properties_widget.get_power()
        # return self.get_property("power")

class InvertNode(NeoAlchemistNode):
    NODE_NAME = "Invert"

    _cache: Optional[npt.ArrayLike] = None

    def __init__(self):
        super().__init__()

        self._in_image = self.define_input("Image")
        self._out_img_port = self.define_output("Image",
            ImageCache(self._handle_request_image_data)
            )

        self._properties_widget = InvertWidget(self.NODE_NAME)

    def _handle_request_image_data(self, roi: ROI):
        return invert(self.in_value("Image").get(roi))

class HueSatNode(NeoAlchemistNode):
    NODE_NAME = "Hue/Saturation"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = HueSatWidget(self.NODE_NAME)

        self.reactive_property("hue", 0,
            self._properties_widget.hue,
            self._properties_widget.set_hue,
            self._properties_widget.hue_changed)

        self.reactive_property("saturation", 1,
            self._properties_widget.saturation,
            self._properties_widget.set_saturation,
            self._properties_widget.saturation_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        # return matrix_sat(in_img, self.get_property("saturation"))
        # return saturation(in_img, self.get_property("saturation"))
        return hue_sat(in_img, self.get_property("hue"), self.get_property("saturation"))

class SaturationNode(NeoAlchemistNode):
    NODE_NAME = "Saturation"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = SaturationWidget(self.NODE_NAME)

        self.reactive_property("saturation", 1,
            self._properties_widget.saturation,
            self._properties_widget.set_saturation,
            self._properties_widget.saturation_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return saturation(in_img, self.get_property("saturation"))

class GammaNode(NeoAlchemistNode):
    NODE_NAME = "Gamma"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = GammaWidget(self.NODE_NAME)

        self.reactive_property("gamma", 1,
            self._properties_widget.gamma,
            self._properties_widget.set_gamma,
            self._properties_widget.gamma_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return gamma(in_img, self.get_property("gamma"))

class ContrastNode(NeoAlchemistNode):
    NODE_NAME = "Contrast"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = ContrastWidget(self.NODE_NAME)

        self.reactive_property("contrast", 1,
            self._properties_widget.contrast,
            self._properties_widget.set_contrast,
            self._properties_widget.contrast_changed)

        self.reactive_property("lift", 0,
            self._properties_widget.lift,
            self._properties_widget.set_lift,
            self._properties_widget.lift_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return linear_contrast(in_img, self.get_property("lift"), self.get_property("contrast"))

# TODO
class EstimateColorBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Estimate Color Balance"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_output("Image")

        self._properties_widget = EstimateColorBalanceWidget(self.NODE_NAME)

# TODO
class PerChannelAverageNode(NeoAlchemistNode):
    NODE_NAME = "Per-Channel Average"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_output("Image")

        self._properties_widget = PerChannelAverageWidget(self.NODE_NAME)
