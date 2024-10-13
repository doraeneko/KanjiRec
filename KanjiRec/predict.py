######################################################################
# KanjiRec
# Helper tool to recognize kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Interface to Kanji prediction
######################################################################

import tf_keras
from tensorflow import keras
from keras import *
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance
from keras.saving import load_model
import PIL
from PIL import ImageOps


model = load_model("%s" % "models/kanji2.keras")
from kanji_lists import JOYO, JINMEIYO


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


def get_prediction(img_path, candidate_count):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(30, 30))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict([img_array])[0]
    result = list(
        LABEL_TO_KANJI_DICT[index]
        for index in np.argsort(prediction)[-candidate_count:]
    )
    print(result)
    return result
