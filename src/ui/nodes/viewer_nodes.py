from typing import Optional

import numpy as np

from ...processing.spells import ImageLike

from ..widgets.PropertiesWidgets import ViewerOutputWidget

from ..widgets.Atoms import ImageRenderer

from .node_basics import NeoAlchemistNode, ROI

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