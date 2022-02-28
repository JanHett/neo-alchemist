import numpy as np

import rawpy
import OpenImageIO as oiio

from ...processing.spells import ImageLike, global_ocio

from ..widgets.PropertiesWidgets import FileOutputWidget, RawFileInputWidget

from .node_basics import NeoAlchemistNode, ImageCache, ROI

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