from typing import Callable, Optional, Tuple, Union
from NodeGraphQt import BaseNode, Port
from PySide2.QtCore import Signal
from numpy.lib.utils import source

import numpy.typing as npt
import numpy as np
import rawpy

from ..processing.spells import ImageFit, ImageLike, fit_image, invert, white_balance

from .Widgets import ColorBalanceWidget, \
    CropWidget, \
    EstimateColorBalanceWidget, \
    FileOutputWidget, \
    GammaWidget, ImageRenderer, \
    InvertWidget, \
    PerChannelAverageWidget, \
    RawFileInputWidget, SolidWidget, \
    TwoPointColorBalanceWidget, \
    ViewerOutputWidget

ORG_IDENTIFIER = "engineering.brotzeit"

def request_data(port):
    # todo: handle data coming from multiple ports?
    source = port.connected_ports()[0]
    return source.request_data()

class ROI:
    def __init__(self,
        position: Tuple[float, float] = (0, 0),
        size: Tuple[float, float] = (1, 1),
        resolution: Tuple[int, int] = (0, 0)) -> None:
        """
        Constructs a region of interest of `size` originating at `position`

        Parameters
        ---
        position
            `[x, y]` position in unit-coordinates `[0..1]`
        size
            `[width, height]` in unit-coordinates `(0..1]`
        resolution
            `[width, height]` in pixels - the special value `0` indicates that
            the original resolution is requested
        """
        self.position = position
        self.size = size
        self.resolution = resolution

    def of(self, img: ImageLike):
        """
        Return the chunk of `img` that is defined by this ROI
        """
        return get_roi(img, self)

    def __eq__(self, other):
        """
        Two ROI are equal if they describe the same area and resolution
        """
        if isinstance(other, ROI):
            return self.position == other.position \
                and self.size == other.size \
                and self.resolution == other.resolution
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self) -> str:
        return f"ROI({self.position}, {self.size}, {self.resolution})"

    def __str__(self) -> str:
        return self.__repr__()

def get_roi(img: ImageLike, roi: ROI) -> ImageLike:
    """
    Return the chunk of `img` that is defined by `roi`
    """
    x, y = np.floor(img.shape[:1] * np.array(roi.position)).astype(int)
    w, h = np.floor(img.shape[:1] * np.array(roi.size)).astype(int)

    sub_img = img[x:x+w, y:y+h]

    # scale to resolution
    sub_img = fit_image(sub_img, roi.resolution)

    return sub_img

class ImageCache:
    SubscriberCallback = Callable[[], None]
    ProviderCallback = Callable[[ROI], ImageLike]

    def __init__(self, provider: ProviderCallback) -> None:
        self._provider = provider
        self._subscribers: set[ImageCache.SubscriberCallback] = set()
        self._cache: dict[ROI, ImageLike] = {}

    def subscribe(self, callback: SubscriberCallback):
        self._subscribers.add(callback)

        callback()

    def unsubscribe(self, callback: SubscriberCallback):
        """
        Remove the subscription of the callback
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get(self, roi: ROI) -> ImageLike:
        return self._get_cache(roi)

    def invalidate_cache(self):
        """
        Clear cache, request fresh values from provider and update all
        subscribers with these
        """
        self._cache = {}
        # call every subscriber to notify them of the new image
        for subscr in self._subscribers:
            subscr()

    def _get_cache(self, roi: ROI):
        """
        Get value for ROI from cache or compute it if it isn't there
        """
        if ROI not in self._cache:
            self._cache[roi] = self._provider(roi)

        return self._cache[roi]

ZERO_CACHE = ImageCache(lambda roi: np.zeros((roi.resolution[1], roi.resolution[0], 3)))

class ReactiveProperty:
    def __init__(self,
        name: str,
        ui_getter: Callable,
        ui_setter: Callable,
        ui_updated_signal: Signal,
        signal_handler: Callable) -> None:
        self.name              = name
        self.ui_getter         = ui_getter
        self.ui_setter         = ui_setter
        self.ui_updated_signal = ui_updated_signal
        self.signal_handler    = signal_handler

class InputHandler:
    def __init__(self, handler: ImageCache.SubscriberCallback) -> None:
        self.handler = handler

class NeoAlchemistNode(BaseNode):
    __identifier__ = ORG_IDENTIFIER

    def __init__(self):
        super().__init__()

        self._reactive_properties: dict[str, ReactiveProperty] = {}
        self._input_handlers: dict[str, InputHandler] = {}
        # TODO: generalise away from ImageCache
        self.out_value: dict[str, ImageCache] = {}

    @property
    def properties_widget(self):
        return self._properties_widget

    def define_output(self, name: str, cache: ImageCache) -> Port:
        out = self.add_output(name)
        self.out_value[name] = cache
        return out

    def define_input(self,
        name: str,
        handler: ImageCache.SubscriberCallback) -> Port:
        input = self.add_input(name)
        self._input_handlers[name] = InputHandler(handler)
        return input

    def in_value(self, name):
        """
        Get the value from the node connected to the input called `name`
        """
        connected = self.get_input(name).connected_ports()
        if len(connected) == 0:
            return ZERO_CACHE
        source_port: Port = connected[0]
        source_cache: ImageCache = source_port.node().out_value[source_port.name()]
        return source_cache

    def input_data(self, input_name):
        return request_data(self.get_input(input_name))

    def reactive_property(self,
        name: str,
        initial_value,
        ui_getter: Callable,
        ui_setter: Callable,
        ui_updated_signal: Signal):
        """
        Defines a node property whose value is linked with that of a UI element

        The type of the property must match the UI element's getter return type
        and setter argument type.
        """
        def signal_handler(*args):
            # this change is coming from the UI, no need to go the complicated
            # way for setting the property
            self._set_property(name, ui_getter())

        ui_updated_signal.connect(signal_handler)

        self._reactive_properties[name] = ReactiveProperty(
            name,
            ui_getter,
            ui_setter,
            ui_updated_signal,
            signal_handler)

        ui_setter(initial_value)
        self.create_property(name, initial_value)

    def _set_property(self, name, value):
        """
        Set property `name` to `value` without updating the UI elements
        associated with reactive properties
        """
        rv = super().set_property(name, value)

        # update ImageCache for all outputs
        for out in self.out_value:
            self.out_value[out].invalidate_cache()

        return rv

    def set_property(self, name, value):
        """
        Set property `name` to `value`
        """
        print(f"Setting {name} to {value}")
        # if this is a reactive property, also set it on the UI widget
        if name in self._reactive_properties:
            prop = self._reactive_properties[name]
            prop.ui_updated_signal.disconnect(prop.signal_handler)
            prop.ui_setter(value)
            prop.ui_updated_signal.connect(prop.signal_handler)

        return self._set_property(name, value)

    def on_input_connected(self, in_port: Port, out_port: Port):
        cache: ImageCache = out_port.node().out_value[out_port.name()]
        input_handler = self._input_handlers[in_port.name()]
        cache.subscribe(input_handler.handler)
        return super().on_input_connected(in_port, out_port)

    def on_input_disconnected(self, in_port: Port, out_port: Port):
        cache: ImageCache = out_port.node().out_value[out_port.name()]
        cache.unsubscribe(self._input_handlers[in_port.name()].handler)
        return super().on_input_disconnected(in_port, out_port)

class SolidNode(NeoAlchemistNode):
    NODE_NAME = "Solid"

    def __init__(self):
        super().__init__()

        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = SolidWidget(self.NODE_NAME)
        self.reactive_property("output_color", self._properties_widget.get_color(),
            self._properties_widget.get_color,
            self._properties_widget.set_color,
            self._properties_widget.color_changed)

    def _handle_request_image_data(self, roi: ROI):
        print(f"Solid:: Returning pixels for {roi}")
        shape = (roi.resolution[1], roi.resolution[0], 3) if roi.resolution[0] > 0 and roi.resolution[1] > 0 else (1, 1, 3)
        return np.full(shape, self._properties_widget.get_color())


class RawFileInputNode(NeoAlchemistNode):
    NODE_NAME = "Raw File Input"

    _raw_data: rawpy.RawPy
    _pixels: ImageLike

    def __init__(self):
        super().__init__()

        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = RawFileInputWidget(self.NODE_NAME)
        self.reactive_property("filename", "",
            self._properties_widget.get_filename,
            self._properties_widget.set_filename,
            self._properties_widget.filename_changed)

    def _handle_request_image_data(self, roi: ROI):
        print(f"Returning pixels for {roi}")
        return roi.of(self._pixels)

    def _set_property(self, name, value):
        if name == "filename":
            print(f"reading data from {value}")
            self._raw_data = rawpy.imread(value)
            self._postprocess_raw_data()

        return super()._set_property(name, value)

    def _postprocess_raw_data(self):
        self._pixels = np.array(self._raw_data.postprocess(
            # TODO: make this conditional on user input or read full size lazily
            half_size=True,
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            no_auto_bright=True
            ), dtype=np.float32) / np.iinfo(np.uint16).max

    @property
    def filename(self) -> str:
        return self.get_property("filename")

class FileOutputNode(NeoAlchemistNode):
    NODE_NAME = "File Output"

    def __init__(self):
        super().__init__()

        self.add_input("Image")

        self._properties_widget = FileOutputWidget(self.NODE_NAME)

def curry_ViewerOutputNode(viewer: ImageRenderer):
    """
    Define a `ViewerOutputNode` for the given `viewer`
    """
    class ViewerOutputNode(NeoAlchemistNode):
        NODE_NAME = "Viewer Output"

        _viewer = viewer

        def __init__(self):
            super().__init__()


            self._properties_widget = ViewerOutputWidget(self.NODE_NAME)

            self.reactive_property("viewer_width", 512,
                self._properties_widget.width_input.value,
                self._properties_widget.width_input.setValue,
                self._properties_widget.width_input.editingFinished)

            self.reactive_property("viewer_height", 512,
                self._properties_widget.height_input.value,
                self._properties_widget.height_input.setValue,
                self._properties_widget.height_input.editingFinished)

            self._in_image = self.define_input("Image",
                self._handle_input_change
                )

        def _set_property(self, name, value):
            rv = super()._set_property(name, value)
            if name == "viewer_width" or name == "viewer_height":
                self._handle_input_change()
            return rv

        def _handle_input_change(self):
            img = self.in_value("Image").get(ROI((0, 0), (1, 1), self.resolution))
            self._update_viewer(img)

        def _update_viewer(self, img: Optional[ImageLike]):
            if img is None:
                # Nothing to show, set image to zero, update viewer and bail
                # TODO: set image to something saying "INPUT MISSING"
                self._viewer.set_image(
                    np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.float32)
                    )
                return

            self._viewer.set_image(img)
            self._viewer.update()

        @property
        def resolution(self):
            return (
                self.get_property("viewer_width"),
                self.get_property("viewer_height")
            )

    return ViewerOutputNode

class CropNode(NeoAlchemistNode):
    NODE_NAME = "Crop"

    def __init__(self):
        super().__init__()

        self.add_input("Image")
        self.define_input("Image", self._handle_input_change)
        self.define_output("Image", ImageCache(self._handle_request_image_data))

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

class GreyBalanceNode(NeoAlchemistNode):
    NODE_NAME = "Grey Balance"
    def __init__(self):
        super().__init__()

        self.add_input("Image")
        image_out = self.add_output("Image")
        image_out.request_data = self._handle_request_image_data

        self._properties_widget = ColorBalanceWidget(self.NODE_NAME)

        self.reactive_property("red", 1,
            self._properties_widget.red.value,
            self._properties_widget.red.setValue,
            self._properties_widget.red.valueChanged)
        self.reactive_property("green", 1,
            self._properties_widget.green.value,
            self._properties_widget.green.setValue,
            self._properties_widget.green.valueChanged)
        self.reactive_property("blue", 1,
            self._properties_widget.blue.value,
            self._properties_widget.blue.setValue,
            self._properties_widget.blue.valueChanged)

    def run(self):
        in_img = request_data(self.get_input("Image"))
        self._cache = white_balance(in_img, self.grey_balance)
        self.update_stream()

    @property
    def grey_balance(self):
        return (
            self.get_property("red"),
            self.get_property("green"),
            self.get_property("blue")
        )

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

        self._in_image = self.define_input("Image",
            self._handle_input_change
            )
        self._out_img_port = self.define_output("Image",
            ImageCache(self._handle_request_image_data)
            )

        self._properties_widget = InvertWidget(self.NODE_NAME)

    def _handle_input_change(self):
        self.out_value["Image"].invalidate_cache()

    def _handle_request_image_data(self, roi: ROI):
        return invert(self.in_value("Image").get(roi))

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
