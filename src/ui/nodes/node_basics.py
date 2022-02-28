from typing import Tuple, Callable

import numpy as np
import numpy.typing as npt

from NodeGraphQt import BaseNode, Port
from PySide2.QtCore import Signal

from ...processing.spells import ImageLike, fit_image

ORG_IDENTIFIER = "engineering.brotzeit"

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
        signal_handler: Callable,
        formatter: Callable) -> None:
        self.name              = name
        self.ui_getter         = ui_getter
        self.ui_setter         = ui_setter
        self.ui_updated_signal = ui_updated_signal
        self.signal_handler    = signal_handler
        self.formatter         = formatter

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
        ui_updated_signal: Signal,
        formatter = lambda v : v):
        """
        Defines a node property whose value is linked with that of a UI element

        The type of the property must match the UI element's getter return type
        and setter argument type.

        Parameters
        ---
        ...

        formatter:
            Function to transform the value before storing it into the property
        """
        self.create_property(name, formatter(initial_value))

        def signal_handler(*args):
            # this change is coming from the UI, no need to go the complicated
            # way for setting the property
            self._set_property(name, formatter(ui_getter()))

        ui_updated_signal.connect(signal_handler)

        self._reactive_properties[name] = ReactiveProperty(
            name,
            ui_getter,
            ui_setter,
            ui_updated_signal,
            signal_handler,
            formatter)

        ui_setter(initial_value)

    def _set_property(self, name, value):
        """
        Set property `name` to `value` without updating the UI elements
        associated with reactive properties
        """
        is_reactive_prop = name in self._reactive_properties

        # format reactive property according to specified formatter
        if is_reactive_prop:
            v = self._reactive_properties[name].formatter(value)
        else:
            v = value

        rv = super().set_property(name, v)

        # only invalidate cache for custom properties
        if is_reactive_prop:
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
