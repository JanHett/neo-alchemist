from typing import Optional, Tuple
from NodeGraphQt import BaseNode, Port

import numpy.typing as npt
import numpy as np
import rawpy

from ..processing.spells import fit_image, invert

from .Widgets import CropWidget, EstimateColorBalanceWidget, FileOutputWidget, GammaWidget, InvertWidget, PerChannelAverageWidget, RawFileInputWidget, TwoPointColorBalanceWidget, ViewerOutputWidget

ORG_IDENTIFIER = "engineering.brotzeit"

def request_data(port):
    # todo: handle data coming from multiple ports?
    source = port.connected_ports()[0]
    return source.request_data()

class NeoAlchemistNode(BaseNode):
    __identifier__ = ORG_IDENTIFIER
    @property
    def properties_widget(self):
        return self._properties_widget

    def on_input_connected(self, source, target):
            self.run()

    def on_input_disconnected(self, source, target):
        self.run()

class RawFileInputNode(NeoAlchemistNode):
    NODE_NAME = "Raw File Input"

    _cache: Tuple[str, Optional[npt.ArrayLike]] = ("", None)

    def __init__(self):
        super().__init__()

        self.add_output("Image")
        self.outputs()["Image"].request_data = self._handle_request_image_data
        self.create_property("filename", "")

        self._properties_widget = RawFileInputWidget(self.NODE_NAME)
        self._properties_widget.filename_changed.connect(self.set_filename)

    def set_filename(self, filename):
        self.set_property("filename", filename)
        self._refresh_cache()

    def _handle_request_image_data(self):
        return self._cache[1]

    def _refresh_cache(self):
        if (self._cache[0] == self.filename and self._cache[1] is not None):
            return

        raw_data = rawpy.imread(self.filename)
        data = np.array(raw_data.postprocess(
            half_size=True, # TODO: make this conditional on user input
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            no_auto_bright=True
            ), dtype=np.float32) / np.iinfo(np.uint16).max

        self._cache = [self.filename, data]

        self.update_stream()

    @property
    def filename(self) -> str:
        return self.get_property("fielname")

class FileOutputNode(NeoAlchemistNode):
    NODE_NAME = "File Output"

    def __init__(self):
        super().__init__()

        self.add_input("Image")

        self._properties_widget = FileOutputWidget(self.NODE_NAME)

def curry_ViewerOutputNode(viewer):
    """
    Define a `ViewerOutputNode` for the given `viewer`
    """
    class ViewerOutputNode(NeoAlchemistNode):
        NODE_NAME = "Viewer Output"

        _viewer = viewer

        def __init__(self):
            super().__init__()

            self.add_input("Image")
            self.create_property("resolution", (0, 0))

            self._properties_widget = ViewerOutputWidget(self.NODE_NAME)
            self._properties_widget.width_input.valueChanged.connect(self._handle_input_change)
            self._properties_widget.width_input.setValue(512)
            self._properties_widget.height_input.valueChanged.connect(self._handle_input_change)
            self._properties_widget.height_input.setValue(512)

            self.run()

        def run(self):
            self._update_cache()

        def _handle_input_change(self):
            self.set_property("resolution",
                (self._properties_widget.width_input.value(),
                self._properties_widget.height_input.value())
            )
            self._update_cache()

        def _update_viewer(self):
            self._viewer.set_image(self._cache[1])

        def _update_cache(self):
            input_port = self.get_input("Image")
            if len(input_port.connected_ports()) == 0:
                # Nothing connected, set image to zero, update viewer and bail
                # TODO: set image to something saying "INPUT MISSING"
                self._cache = (
                    f"zeros{self.resolution}",
                    np.zeros((*self.resolution, 3), dtype=np.float32)
                    )
                self._update_viewer()
                return

            source_port: Port = input_port.connected_ports()[0]
            source_identifier = source_port.node().id + source_port.name() + str(self.resolution)

            self._cache = (
                source_identifier,
                fit_image(request_data(input_port), self.resolution)
                )

            # show new image in viewer
            self._update_viewer()

        @property
        def resolution(self):
            return self.get_property("resolution")

    return ViewerOutputNode

class CropNode(NeoAlchemistNode):
    NODE_NAME = "Crop"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        image_out = self.add_output("Image")
        image_out.request_data = self._handle_request_image_data

        self._properties_widget = CropWidget(self.NODE_NAME)


    @property
    def crop_window(self):
        # TODO: implement widget, then get actual values
        x      = 1 # self._properties_widget.x.value()
        y      = 1 # self._properties_widget.y.value()
        width  = 1 # self._properties_widget.width.value()
        height = 1 # self._properties_widget.height.value()

        return (x, y, width, height)

    def run(self):
        in_img = request_data(self.get_input("Image"))

        w = in_img.shape[0]
        h = in_img.shape[1]

        y1 = int(self.crop_window[1] * h)
        y2 = int((self.crop_window[1] + self.crop_window[3]) * h)

        x1 = int(self.crop_window[0] * w)
        x2 = int((self.crop_window[0] + self.crop_window[2]) * w)

        self._cache = in_img[y1:y2, x1:x2]

    def _handle_request_image_data(self):
        return self._cache

class TwoPointColorBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Two-Point Color Balance"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_input("Light Sample")
        self.add_input("Dark Sample")
        self.add_output("Image")

        self._properties_widget = TwoPointColorBalanceWidget(self.NODE_NAME)

class InvertNode(NeoAlchemistNode):
    NODE_NAME = "Invert"

    _cache: Optional[npt.ArrayLike] = None

    def __init__(self):
        super().__init__()

        self._in_img_port  = self.add_input("Image")
        self._out_img_port = self.add_output("Image")
        self._out_img_port.request_data = self._handle_request_image_data

        self._properties_widget = InvertWidget(self.NODE_NAME)

    def run(self):
        in_img = request_data(self._in_img_port)

        self._cache = invert(in_img)

    def _handle_request_image_data(self):
        if self._cache is None:
            self.run()
        return self._cache

class GammaNode(NeoAlchemistNode):
    NODE_NAME = "Gamma"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_output("Image")

        self._properties_widget = GammaWidget(self.NODE_NAME)

class EstimateColorBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Estimate Color Balance"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_output("Image")

        self._properties_widget = EstimateColorBalanceWidget(self.NODE_NAME)

class PerChannelAverageNode(NeoAlchemistNode):
    NODE_NAME = "Per-Channel Average"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.add_output("Image")

        self._properties_widget = PerChannelAverageWidget(self.NODE_NAME)
