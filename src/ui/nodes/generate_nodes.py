import numpy as np

from ..widgets.PropertiesWidgets import SolidWidget

from .node_basics import NeoAlchemistNode, ImageCache, ROI

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