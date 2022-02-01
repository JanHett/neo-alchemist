################################################################################
# Entry point to the Graphical interface of the Neo Alchemist
################################################################################

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDockWidget, QMainWindow

from NodeGraphQt import NodeGraph, Port, BackdropNode, setup_context_menu

from .widgets.Atoms import ViewerWidget
from .Nodes import \
    SolidNode, \
    RawFileInputNode, \
    FileOutputNode, \
    curry_ViewerOutputNode, \
    ColorSpaceTransformNode, \
    CropNode, \
    GreyBalanceNode, \
    TwoPointColorBalanceNode, \
    CDLNode, \
    InvertNode, \
    HueSatNode, \
    SaturationNode, \
    GammaNode, \
    ContrastNode, \
    EstimateColorBalanceNode, \
    PerChannelAverageNode, \
    AddNode, \
    MultiplyNode, \
    EqualsNode, \
    LessThanNode, \
    GreaterThanNode, \
    OrNode, \
    AndNode

class MainWindow(QMainWindow):
    def __init__(self, cl_args) -> None:
        super().__init__()

        self.setWindowTitle("Neo Alchemist")
        
        #cl_args.file
        viewer = ViewerWidget(self)

        self.setCentralWidget(viewer)

        self._nodes = [
            # BackdropNode,
            SolidNode,
            RawFileInputNode,
            FileOutputNode,
            curry_ViewerOutputNode(viewer.image_renderer),
            ColorSpaceTransformNode,
            CropNode,
            GreyBalanceNode,
            TwoPointColorBalanceNode,
            CDLNode,
            InvertNode,
            HueSatNode,
            SaturationNode,
            GammaNode,
            ContrastNode,
            EstimateColorBalanceNode,
            PerChannelAverageNode,
            AddNode,
            MultiplyNode,
            EqualsNode,
            LessThanNode,
            GreaterThanNode,
            OrNode,
            AndNode
        ]

        self.create_node_container()
        self.create_properties_panel()
        self.create_favorite_properties_panel()
        if cl_args.file:
            self.open_file(cl_args.file)

    def register_node(self, node):
        """
        Add `node` to the list of available node types
        """
        self._nodes.append(node)
        self._node_graph.register_node(node)

    def create_node_container(self):
        self._node_container = QDockWidget("Recipe", self)

        self._node_graph = NodeGraph(self._node_container)
        self._node_graph.use_OpenGL()
        setup_context_menu(self._node_graph)

        for n in self._nodes:
            self._node_graph.register_node(n)

        graph_widget = self._node_graph.widget
        self._node_container.setWidget(graph_widget)

        self.addDockWidget(Qt.BottomDockWidgetArea, self._node_container)

        self._node_graph.node_double_clicked.connect(self._handle_node_click)

    def create_properties_panel(self):
        self._properties_container = QDockWidget("Properties", self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self._properties_container)

    def create_favorite_properties_panel(self):
        self._favorite_properties_container = QDockWidget("Favorite Properties", self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self._favorite_properties_container)

    def open_file(self, filename: str):
        self._node_graph.load_session(filename)

    def _handle_node_click(self, node):
        self._properties_container.setWidget(node.properties_widget)
