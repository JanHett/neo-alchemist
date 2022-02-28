from ..widgets.PropertiesWidgets import AddWidget, AndWidget, EqualsWidget, GreaterThanWidget, LessThanWidget, MultiplyWidget, OrWidget

from .node_basics import NeoAlchemistNode, ImageCache, ROI

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
