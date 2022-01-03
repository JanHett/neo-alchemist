from typing import Callable, Optional, Tuple, Union
from PySide2.QtWidgets import QGroupBox, QWidget
import numpy.typing as npt
from ..processing.spells import invert, white_balance

from ..ui.Widgets import ColorBalanceWidget, FileIOWidget, InvertWidget, PipelineWidget

class Controller:
    """
    Combines UI and processing logic into a resuable building block
    """
    def __init__(self, headless: bool) -> None:
        self._headless = headless
        if not headless:
            self.widget: Optional[QWidget] = None

    def _process_impl(self, image: npt.ArrayLike) -> npt.ArrayLike:
        return image

    def process(self, image: npt.ArrayLike) -> npt.ArrayLike:
        """
        Execute the processing step described by this controller using the 
        currently set parameters
        
        Inheriting classes should override `_process_impl` to define the
        processing step.
        """
        try:
            return self._process_impl(image)
        except Exception as e:
            # TODO: show error message in UI and/or on the terminal
            print(e)

    def on_updated(self, image: npt.ArrayLike, callback: Callable[[npt.ArrayLike], None]):
        """
        Whenever there is an update available, `image` will be processed and
        `callback` will be passed the result as an argument

        Parameters
        ---

        image:
            A float32 image to white balance
        callback:
            a unary function taking a float32 image as an argument
        """
        self._image = image
        self._callback = callback

    def _process_update(self):
        try:
            self._callback(self.process(self._image))
        except AttributeError:
            print("Cannot process: image or callback may be missing.")

class PipelineController(Controller):
    def __init__(self, headless: bool,
        parent: Optional[Union[Controller, QWidget]] = None) -> None:
        super().__init__(headless)
        if isinstance(parent, Controller):
            self._parent: Controller = parent

        self._pipeline: list[Controller] = []

        if not self._headless:
            p = parent.widget if isinstance(parent, Controller) else parent
            self.widget = PipelineWidget(p)

    @property
    def pipeline(self):
        return self._pipeline

    def push_step(self, s: Controller):
        self._pipeline.append(s)
        if not self._headless:
            self.widget.push_step_widget(s.widget)

    def process(self, image: Optional[npt.ArrayLike]) -> npt.ArrayLike:
        try:
            return self._process_impl(image)
        except Exception as e:
            # TODO: show error message in UI and/or on the terminal
            print(e)

    def _process_impl(self, image: Optional[npt.ArrayLike]):
        if isinstance(self._pipeline[0], FileInputController):
            temp = self._pipeline[0].image
        else:
            # TODO: handle case where first controller isn't FileInput and there
            # is no argument
            temp = image

        for s in self._pipeline:
            temp = s.process(temp)

        return temp

class InvertController(Controller):
    def __init__(self,
        headless: bool = False,
        default_activated: bool = True,
        title: str = "",
        parent: Optional[Controller] = None):
        """
        Construct an InvertController
        
        Parameters
        ---
        
        headless
            If `True` the Controller is initialised without UI
        default_activated
            A setting to initialise the controller with
        title
            A description to display in the widget
        parent
            If this Controller is not run `headless`, its widget will be
            parented to `parent.widget`
        """
        super().__init__(headless)
        self._parent = parent
        self._default_activated = default_activated

        if not headless:
            self.widget = InvertWidget(title, self._parent.widget)

            self.widget.activated.setChecked(self._default_activated)
            self.widget.activated.stateChanged.connect(self._process_update)

    @property
    def activated(self):
        if self._headless:
            return self._default_activated
        return self.widget.activated.value()

    def _process_impl(self, image: npt.ArrayLike):
        if self.activated:
            return invert(image)
        return image

class WhiteBalanceController(Controller):
    def __init__(self,
        headless: bool = False,
        default_activated: bool = True,
        default_white_balance: Tuple[float, float, float] = (1, 1, 1),
        title: str = "",
        parent: Optional[Controller] = None):
        """
        Construct a WhiteBalanceController
        
        Parameters
        ---
        
        headless
            If `True` the Controller is initialised without UI
        default_activated
            A setting to initialise the controller with
        default_white_balance
            A setting to initialise the controller with
        title
            A description to display in the widget
        parent
            If this Controller is not run `headless`, its widget will be
            parented to `parent.widget`
        """
        super().__init__(headless)
        self._parent = parent
        self._default_activated = default_activated
        self._default_white_balance = default_white_balance

        if not headless:
            self.widget = ColorBalanceWidget(title, self._parent.widget)

            self.widget.activated.setChecked(self._default_activated)
            self.widget.activated.stateChanged.connect(self._process_update)

            self.widget.red.setValue(self._default_white_balance[0])
            self.widget.green.setValue(self._default_white_balance[1])
            self.widget.blue.setValue(self._default_white_balance[2])

            self.widget.red.valueChanged.connect(self._process_update)
            self.widget.green.valueChanged.connect(self._process_update)
            self.widget.blue.valueChanged.connect(self._process_update)

    @property
    def white_balance(self):
        return self._default_white_balance if self._headless else (
            self.widget.red.value(),
            self.widget.green.value(),
            self.widget.blue.value(),
            )

    @property
    def activated(self):
        if self._headless:
            return self._default_activated
        return self.widget.activated.value()

    def _process_impl(self, image: npt.ArrayLike):
        if self.activated:
            return white_balance(image, self.white_balance)
        return image

class FileInputController(Controller):
    def __init__(self, headless: bool,
        title: str = "",
        parent: Optional[Controller] = None) -> None:
        super().__init__(headless)

        self._parent = parent

        if not headless:
            self.widget = FileIOWidget(title, self._parent.widget, "open")

class FileOutputController(Controller):
    def __init__(self, headless: bool,
        title: str = "",
        parent: Optional[Controller] = None) -> None:
        super().__init__(headless)

        self._parent = parent

        if not headless:
            self.widget = FileIOWidget(title, self._parent.widget, "save")

class SaturationController(Controller):
    pass
