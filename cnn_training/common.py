######################################################################
# KanjiRec
# Helper tool to recognize Kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Create training data sets for Kanji recognizer
######################################################################

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from kanji_lists import JOYO, JINMEIYO
from pathlib import Path
import os


def generate_kanji_dictionary():

    # Create a dictionary to store kanjis and their corresponding numbers
    kanji_dict = {}
    reverse_kanji_dict = {}
    label_to_kanji_dict = {}
    counter = 1
    kanjis = []
    for kanji in JOYO.HEISEI22:
        kanjis.append(kanji)
    for kanji in JINMEIYO.HEISEI29:
        kanjis.append(kanji)
    for kanji in sorted(kanjis):
        kanji_dict[kanji] = counter
        reverse_kanji_dict[counter] = kanji
        counter += 1
    numbers_as_strings = sorted(list(str(x) for x in kanji_dict.values()))
    counter = 0
    for number in numbers_as_strings:
        label_to_kanji_dict[counter] = reverse_kanji_dict[int(number)]
        counter += 1
    return (kanji_dict, reverse_kanji_dict, label_to_kanji_dict)


IMAGE_SCALE = 30
KANJI_DICT, REV_KANJI_DICT, LABEL_TO_KANJI_DICT = generate_kanji_dictionary()
TRANING_DATA_DIR = "training_data/data"
