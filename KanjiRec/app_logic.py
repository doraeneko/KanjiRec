######################################################################
# KanjiRec
# Helper tool to recognize Kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Application logic (i.e., resource manipulation) is done in
# AppLogic. The state of the
# application is stored in resource AppState.
######################################################################

import enum
import os
import pickle
import asyncio

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QPoint, QRect

try:
    from .resources import Resource, Resources
    from .predict import get_prediction

except:
    from resources import Resource, Resources
    from predict import get_prediction

class AppState(enum.Enum):
    LOADING_AND_PREPARING = 0
    INITIAL = 1
    IMAGE_COPIED_NO_MARKING = 2
    MARKING_GIVEN = 3


class AppLogic:
    def __init__(self):
        resources_dict = {
            "AppState": AppState.LOADING_AND_PREPARING,
            "Image": None,
            "OriginalClipboardImage": None,
            "Marking": None,
            "MarkingImage": None,
            "Predictions": None,
        }
        self._r = Resources(resources_dict)
        self._r.add_listener(
            "OriginalClipboardImage", self.on_change_original_clipboard_image
        )
        self._r.add_listener("Marking", self.on_change_marking)

    def get_resources(self):
        return self._r

    def get_state(self):
        return self._r.AppState

    def on_change_original_clipboard_image(self):
        if self._r.AppState == AppState.LOADING_AND_PREPARING:
            return
        self._r.AppState = AppState.INITIAL
        if self._r.OriginalClipboardImage:
            self._r.AppState = AppState.IMAGE_COPIED_NO_MARKING
        else:
            # jump to initial state
            self._r.AppState = AppState.INITIAL

    def on_change_marking(self):
        if not self._r.Marking:
            return

        point1: QPoint = self._r["Marking"][0]
        point2: QPoint = self._r["Marking"][1]
        # self._r["Marking"] = None
        x1 = min(point1.x(), point2.x())
        y1 = min(point1.y(), point2.y())
        x2 = max(point1.x(), point2.x())
        y2 = max(point1.y(), point2.y())
        rect = QRect(x1, y1, x2 - x1, y2 - y1)
        pixmap: QPixmap = self._r["OriginalClipboardImage"]
        subimage = pixmap.copy(rect)
        scaled_subimage = subimage.scaled(30, 30)
        img = scaled_subimage.toImage()
        greyscale_image = img.convertToFormat(QImage.Format.Format_Grayscale8)
        greyscale_image.save("marking.png")
        self._r["MarkingImage"] = "marking.png"
        self._r["Predictions"] = get_prediction(self._r["MarkingImage"], 5)
        if self._r.Marking and self._r.AppState in [AppState.IMAGE_COPIED_NO_MARKING]:
            self._r.AppState = AppState.MARKING_GIVEN
