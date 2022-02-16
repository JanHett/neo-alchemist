import sys
from src.ui.MainWindow import MainWindow
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication
import argparse

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert a raw image of a colour negative to a positive")
    parser.add_argument("--file", "-f",
        type=str,
        required=False,
        help="A path to a Neo Alchemist recipe file (*.json)")

    args = parser.parse_args()

    window = MainWindow(args)
    window.show()
    sys.exit(app.exec())
