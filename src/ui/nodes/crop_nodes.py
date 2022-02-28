from ..widgets.PropertiesWidgets import CropWidget

from .node_basics import NeoAlchemistNode, ImageCache

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
        in_img = [] # request_data(self.get_input("Image"))

        w = in_img.shape[0]
        h = in_img.shape[1]

        y1 = int(self.crop_window[1] * h)
        y2 = int((self.crop_window[1] + self.crop_window[3]) * h)

        x1 = int(self.crop_window[0] * w)
        x2 = int((self.crop_window[0] + self.crop_window[2]) * w)

        self._cache = in_img[y1:y2, x1:x2]

    def _handle_request_image_data(self):
        return self._cache