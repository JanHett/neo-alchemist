import sys
from src.ui.MainWindow import MainWindow
from PySide6.QtWidgets import QApplication
import argparse

if __name__ == "__main__":
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Convert a raw image of a colour negative to a positive")
    parser.add_argument("file", type=str, help="The path to the digitised negative to process")

    args = parser.parse_args()

    window = MainWindow(args)
    window.show()
    sys.exit(app.exec())
