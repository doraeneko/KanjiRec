######################################################################
# KanjiRec
# Helper tool to recognize Kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Main window containing all controls.
######################################################################


from PyQt6.QtWidgets import QApplication
import sys

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    from .clipboard_image_widget import ClipboardImageWidget
    from .app_logic import AppState
    from .app_logic import AppState, AppLogic
    from .predict import get_prediction
except:
    from clipboard_image_widget import ClipboardImageWidget
    from app_logic import AppState, AppLogic
    from app_logic import AppState, AppLogic
    from predict import get_prediction

class MainWindow(QMainWindow):
    """Main window of the add-on."""

    def __init__(self):
        super().__init__()
        self._app_logic = AppLogic()
        self._r = self._app_logic.get_resources()
        self.build_gui()
        self.add_listeners()
        #self.hide()
        self._r["AppState"] = AppState.INITIAL

    def reset_choices(self, predictions = None):
        for index, edit in enumerate(self._choices):
            assert(isinstance(edit, QTextEdit))
            edit.setReadOnly(True)
            edit.setFixedSize(50, 50)
            edit.setText("%s" % predictions[index] if predictions else "?")

    def build_gui(self):
        self.setWindowTitle("KanjiRec")
        self._central_widget = QWidget()
        self._default_font = QFont()
        self._default_font.setPointSize(14)
        self._central_widget.setFont(self._default_font)
        self.setCentralWidget(self._central_widget)
        self._layout = QVBoxLayout()
        self._central_widget.setLayout(self._layout)
        canvas_layout = QGridLayout()
        canvas_box = QGroupBox("Screenshot and marking:")
        canvas_size = 350
        canvas_box_size = canvas_size + 50
        canvas_box.setFixedSize(canvas_box_size, canvas_box_size)
        canvas_box.setMaximumHeight(canvas_box_size)
        canvas_box.setMinimumHeight(canvas_box_size)
        canvas_box.setMaximumWidth(canvas_box_size)
        canvas_box.setMinimumWidth(canvas_box_size)
        self._canvas_box = canvas_box
        self._canvas = ClipboardImageWidget(self, self._r, canvas_size, canvas_size)
        self._canvas.setFixedSize(canvas_size, canvas_size)
        canvas_layout.addWidget(
            self._canvas, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter
        )
        canvas_box.setLayout(canvas_layout)
        self._layout.addWidget(canvas_box)
        self._choices = [
            QTextEdit(),
            QTextEdit(),
            QTextEdit(),
            QTextEdit(),
            QTextEdit(),
        ]
        self.reset_choices()
        hbox_layout = QHBoxLayout()
        for btn in self._choices:
            hbox_layout.addWidget(btn)
        self._layout.addLayout(hbox_layout)
        self._status_label = QLabel()
        self.statusBar().addWidget(self._status_label)
        self.setFixedSize(self.minimumSizeHint())
        self.update_status_for_gui_controls()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

    def add_listeners(self):
        self._r.add_listener("AppState", self.update_status_for_gui_controls)
        self._r.add_listener("Predictions", self.update_status_for_gui_controls)

    def update_status_for_gui_controls(self):
        if self._r["AppState"] == AppState.LOADING_AND_PREPARING:
            return
        self._in_process_of_state_updating = True
        current_state = self._r["AppState"]
        if current_state == AppState.INITIAL:
            self._canvas.setDisabled(False)
            self.set_status_message(
                'Copy a picture to the clipboard, e.g using "Snipping" tool.'
            )
        elif current_state == AppState.IMAGE_COPIED_NO_MARKING:
            self._canvas.setDisabled(False)
            self.set_status_message("Mark the kanji in the picture using the mouse.")
        elif current_state == AppState.MARKING_GIVEN:
            self._canvas.setDisabled(False)
            self.set_status_message("Computed top-5 predictions.")
        else:
            self.set_status_message("Unknown state.")
        self._in_process_of_state_updating = False
        self.reset_choices(self._r["Predictions"])

    def set_status_message(self, message):
        self._status_label.setText(message)

    def on_entry_changed(self):
        if self._in_process_of_state_updating:
            return
        self._r["CurrentEntry"] = self._entry_edit.text()



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
