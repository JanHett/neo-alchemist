import os.path
from typing import Callable, Optional, Tuple, Union

from NodeGraphQt import BaseNode, Port
from PySide2.QtCore import Signal

import numpy.typing as npt
import numpy as np

import rawpy
import OpenImageIO as oiio
import lcms

from ..processing.spells import ImageFit, \
    ImageLike, cdl, \
    fit_image, \
    gamma, \
    hue_sat, \
    invert, \
    global_ocio, \
    linear_contrast, \
    matrix_sat, \
    saturation, \
    two_point_color_balance, \
    white_balance

from .widgets.PropertiesWidgets import AddWidget, AndWidget, CDLWidget, ColorBalanceWidget, ColorSpaceTransformWidget, ContrastWidget, \
    CropWidget, EqualsWidget, \
    EstimateColorBalanceWidget, \
    FileOutputWidget, \
    GammaWidget, GreaterThanWidget, HueSatWidget, \
    InvertWidget, LessThanWidget, MultiplyWidget, OrWidget, \
    PerChannelAverageWidget, \
    RawFileInputWidget, SaturationWidget, SolidWidget, \
    TwoPointColorBalanceWidget, \
    ViewerOutputWidget

from .widgets.Atoms import ImageRenderer

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

    def __ne__(self, other):
        return not(self == other)

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
    if 0 in roi.resolution:
        resolution = img.shape[:2]
        resolution = (resolution[1], resolution[0])
    else:
        resolution = roi.resolution

    y, x = np.floor(
        img.shape[:2] * np.array((roi.position[1], roi.position[0]))
    ).astype(int)
    h, w = np.floor(
        img.shape[:2] * np.array((roi.size[1], roi.size[0]))
    ).astype(int)

    sub_img = img[y:y+h, x:x+w]

    # scale to resolution
    sub_img = fit_image(sub_img, resolution)

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
        if roi not in self._cache:
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
        self._out_values: dict[str, ImageCache] = {}

    @property
    def properties_widget(self):
        return self._properties_widget

    def define_output(self, name: str, cache: ImageCache) -> Port:
        out = self.add_output(name)
        self._out_values[name] = cache
        return out

    def define_input(self,
        name: str,
        handler: ImageCache.SubscriberCallback = None) -> Port:
        """
        Define an input

        The default `handler` invalidates all output caches whenever there is a
        change to the input
        """
        if handler is None:
            handler = self.invalidate_all_output_caches
        input = self.add_input(name)
        self._input_handlers[name] = InputHandler(handler)
        return input

    @property
    def out_values(self):
        return self._out_values

    def out_value(self, name):
        return self._out_values[name]

    def in_value(self, name):
        """
        Get the value from the node connected to the input called `name`
        """
        connected = self.get_input(name).connected_ports()
        if len(connected) == 0:
            return ZERO_CACHE
        source_port: Port = connected[0]
        source_cache: ImageCache = source_port.node().out_value(source_port.name())
        return source_cache

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

        # only invalidate cache for custom properties
        if name not in self.model.properties.keys():
            # update ImageCache for all outputs
            for out in self._out_values.values():
                out.invalidate_cache()

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
        cache: ImageCache = out_port.node().out_value(out_port.name())
        input_handler = self._input_handlers[in_port.name()]
        cache.subscribe(input_handler.handler)
        return super().on_input_connected(in_port, out_port)

    def on_input_disconnected(self, in_port: Port, out_port: Port):
        cache: ImageCache = out_port.node().out_value(out_port.name())
        cache.unsubscribe(self._input_handlers[in_port.name()].handler)
        return super().on_input_disconnected(in_port, out_port)

    def is_input_connected(self, input: str):
        """Returns true if the `input` is connected to an output"""
        port: Port = self.get_input(input)
        return len(port.connected_ports()) > 0

    def invalidate_all_output_caches(self):
        """
        Invalidates the cache of all outputs

        This serves as the default handler for changes in input ImageCache(s)
        """
        for out in self._out_values.values():
            out.invalidate_cache()

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
        if roi.resolution[0] > 0 and roi.resolution[1] > 0:
            shape = (roi.resolution[1], roi.resolution[0], 3)
        else:
            shape = (1, 1, 3)

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
            # output_color=rawpy.ColorSpace.raw,
            # output_color=rawpy.ColorSpace.ProPhoto,
            output_color=rawpy.ColorSpace.XYZ,
            output_bps=16,
            gamma=(1, 1),
            user_wb=[1.0, 1.0, 1.0, 1.0],
            no_auto_bright=True
            ), dtype=np.float32) / np.iinfo(np.uint16).max

        proc = global_ocio.get_processor("Utility - Linear - RIMM ROMM (ProPhoto)",
            "Role - scene_linear")

        proc.applyRGB(self._pixels)

        # self._pixels = self._pixels ** 0.555555

    @property
    def filename(self) -> str:
        return self.get_property("filename")

class FileOutputNode(NeoAlchemistNode):
    NODE_NAME = "File Output"

    def __init__(self):
        super().__init__()

        self.define_input("Image")

        self._properties_widget = FileOutputWidget(self.NODE_NAME)

        self.reactive_property("filename", "",
            self._properties_widget.get_filename,
            self._properties_widget.set_filename,
            self._properties_widget.filename_changed)

        self._properties_widget.process_button.clicked.connect(self._process)

    def _process(self):
        img = self.in_value("Image").get(ROI())
        pixformat = "uint16" # or "uint8"

        # Convert from linear ACES to sRGB (others are TODO)
        # dtype = np.uint16 if pixformat == "uint16" else np.uint8
        # max_val = np.iinfo(dtype).max
        # formatted_pixels = (img.clip(0, 1) * max_val).astype(dtype)
        # this_dir = os.path.abspath(os.path.dirname(__file__))
        # aces_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/ACES-elle-V2-g10.icc")
        # srgb_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/sRGB-elle-V2-srgbtrc.icc")
        # srgb_pixels = lcms.apply_profile(formatted_pixels, aces_profile_path, srgb_profile_path)
        
        # srgb_pixels = lin_to_srgb(img)
        dtype = np.uint16 if pixformat == "uint16" else np.uint8
        max_val = np.iinfo(dtype).max
        srgb_pixels = (img.clip(0, 1) * max_val).astype(dtype)

        filename = self.get_property("filename")
        output = oiio.ImageOutput.create(filename)
        if not output:
            print("OIIO Error". oiio.geterror())
            return

        spec = oiio.ImageSpec(
            srgb_pixels.shape[1],
            srgb_pixels.shape[0],
            srgb_pixels.shape[2],
            pixformat)
        output.open(filename, spec)
        output.write_image(srgb_pixels)
        output.close()

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

class ColorSpaceTransformNode(NeoAlchemistNode):
    NODE_NAME = "Color Space Transform"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = ColorSpaceTransformWidget(self.NODE_NAME, global_ocio.get_colorspaces())

        self.reactive_property("from_space", "default",
            self._properties_widget.get_from_space,
            self._properties_widget.set_from_space,
            self._properties_widget.from_space_changed)

        self.reactive_property("to_space", "default",
            self._properties_widget.get_to_space,
            self._properties_widget.set_to_space,
            self._properties_widget.to_space_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi).copy().astype(np.float32)

        proc = global_ocio.get_processor(self.get_property("from_space"),
            self.get_property("to_space"))

        proc.applyRGB(in_img)

        return in_img

class ComvertICCProfileNode(NeoAlchemistNode):
    NODE_NAME = "Convert ICC Profile"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = ComvertICCProfileWidget(self.NODE_NAME)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)

        # Convert from linear ACES to sRGB (others are TODO)
        this_dir = os.path.abspath(os.path.dirname(__file__))
        aces_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/ACES-elle-V2-g10.icc")
        srgb_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/sRGB-elle-V2-srgbtrc.icc")
        srgb_pixels = lcms.apply_profile(in_img, aces_profile_path, srgb_profile_path)

        # srgb_pixels = lin_to_srgb(in_img)

        return srgb_pixels


# TODO
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
        self._properties_widget.balance_control.update()
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

        self.reactive_property("slope", np.array((1, 1, 1)),
            self._properties_widget.get_slope,
            self._properties_widget.set_slope,
            self._properties_widget.slope_changed)

        self.reactive_property("offset", np.array((0, 0, 0)),
            self._properties_widget.get_offset,
            self._properties_widget.set_offset,
            self._properties_widget.offset_changed)

        self.reactive_property("power", np.array((1, 1, 1)),
            self._properties_widget.get_power,
            self._properties_widget.set_power,
            self._properties_widget.power_changed)

    def _handle_request_image_data(self, roi: ROI):
        self._properties_widget.balance_control.update()
        return cdl(self.in_value("Image").get(roi),
            self.slope, self.offset, self.power)

    @property
    def slope(self):
        return self.get_property("slope")

    @property
    def offset(self):
        return self.get_property("offset")

    @property
    def power(self):
        return self.get_property("power")

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

class AddNode(NeoAlchemistNode):
    NODE_NAME = "Add"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = AddWidget(self.NODE_NAME)

        self.reactive_property("summand", 1,
            self._properties_widget.get_summand,
            self._properties_widget.set_summand,
            self._properties_widget.summand_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return in_img + self.get_property("summand")

class MultiplyNode(NeoAlchemistNode):
    NODE_NAME = "Multiply"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = MultiplyWidget(self.NODE_NAME)

        self.reactive_property("factor", 1,
            self._properties_widget.get_factor,
            self._properties_widget.set_factor,
            self._properties_widget.factor_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return in_img * self.get_property("factor")

class EqualsNode(NeoAlchemistNode):
    NODE_NAME = "Equals"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = EqualsWidget(self.NODE_NAME)

        self.reactive_property("comparison", 1,
            self._properties_widget.get_comparison,
            self._properties_widget.set_comparison,
            self._properties_widget.comparison_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return in_img > self.get_property("comparison")

class LessThanNode(NeoAlchemistNode):
    NODE_NAME = "LessThan"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = LessThanWidget(self.NODE_NAME)

        self.reactive_property("threshold", 1,
            self._properties_widget.get_threshold,
            self._properties_widget.set_threshold,
            self._properties_widget.threshold_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return in_img < self.get_property("threshold")

class GreaterThanNode(NeoAlchemistNode):
    NODE_NAME = "GreaterThan"

    def __init__(self):
        super().__init__()

        self.define_input("Image")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = GreaterThanWidget(self.NODE_NAME)

        self.reactive_property("threshold", 1,
            self._properties_widget.get_threshold,
            self._properties_widget.set_threshold,
            self._properties_widget.threshold_changed)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)
        return in_img > self.get_property("threshold")

class OrNode(NeoAlchemistNode):
    NODE_NAME = "Or"

    def __init__(self):
        super().__init__()

        self.define_input("Image A")
        self.define_input("Image B")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = OrWidget(self.NODE_NAME)

    def _handle_request_image_data(self, roi: ROI):
        in_img_a = self.in_value("Image A").get(roi)
        in_img_b = self.in_value("Image B").get(roi)
        if in_img_a.shape != in_img_b.shape:
            raise ValueError("Boolean operators must be called on images of equal shape")
        return in_img_a or in_img_b

class AndNode(NeoAlchemistNode):
    NODE_NAME = "And"

    def __init__(self):
        super().__init__()

        self.define_input("Image A")
        self.define_input("Image B")
        self.define_output("Image", ImageCache(self._handle_request_image_data))

        self._properties_widget = AndWidget(self.NODE_NAME)

    def _handle_request_image_data(self, roi: ROI):
        in_img_a = self.in_value("Image A").get(roi)
        in_img_b = self.in_value("Image B").get(roi)
        if in_img_a.shape != in_img_b.shape:
            raise ValueError("Boolean operators must be called on images of equal shape")
        return in_img_a and in_img_b
