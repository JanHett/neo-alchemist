import os.path

import numpy as np

from ...processing.spells import global_ocio

from ..widgets.PropertiesWidgets import ColorSpaceTransformWidget

from .node_basics import NeoAlchemistNode, ImageCache, ROI

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
        # TODO: seems to be overwriting despite copy
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

        self._properties_widget = ConvertICCProfileWidget(self.NODE_NAME)

    def _handle_request_image_data(self, roi: ROI):
        in_img = self.in_value("Image").get(roi)

        # Convert from linear ACES to sRGB (others are TODO)
        this_dir = os.path.abspath(os.path.dirname(__file__))
        aces_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/ACES-elle-V2-g10.icc")
        srgb_profile_path = os.path.join(this_dir, "../../external/elles_icc_profiles/profiles/sRGB-elle-V2-srgbtrc.icc")
        srgb_pixels = lcms.apply_profile(in_img, aces_profile_path, srgb_profile_path)

        # srgb_pixels = lin_to_srgb(in_img)

        return srgb_pixels
