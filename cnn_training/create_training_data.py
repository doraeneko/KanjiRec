######################################################################
# KanjiRec
# Helper tool to recognize Kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Create training data sets for Kanji recognizer
######################################################################

import functools
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from kanji_lists import JOYO
from pathlib import Path
import os

from tensorflow_datasets.core.features.text_feature import (
    _file_name_prefix_for_metadata,
)

import common


kanji_dict = common.KANJI_DICT
FONT_DIR = "training_data/fonts"
FONTS = [
    # ("mikiyu", 1.0),
    # ("Makinas-Scrap-5", 1.0),
    ("IPAMinchoRegular", 1.1),
    # ("IPAGothicRegular", 1.1),
    # ("GenJyuuGothicMonospaceHeavy", 1.2),
    # ("GenJyuuGothicMonospaceBold", 1.1),
    # ("GenShinGothicMonospaceBold", 1.1),
    # ("osaka.unicode", 1.0),
    # ("ys_handy_writing", 1.0),
    # ("togoshi-mincho",1.1),
    # ("migu-1p-bold", 1.1),
    # ("migu-1p-regular", 1.0),
    # ("ArialUnicodeMS", 1.0),
]


def add_noise(image: Image, ratio):
    if ratio == 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(ratio))


def create_kanji_image(
    kanji,
    image_size,
    scale_size,
    font_path,
    delta_x,
    delta_y,
    adjustment_scale,
    blurr_factor,
):
    # Create a blank image
    image = Image.new("L", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=int(image_size / adjustment_scale))

    # Calculate the position to center the kanji in the image
    text_width, text_height = draw.textsize(kanji, font=font)
    x = (image_size - text_width) / scale_size + delta_x
    y = (image_size - text_height) / scale_size + delta_y

    if "ArialUnicodeMS" in font_path:
        draw.text(
            (x, y), kanji, font=font, fill="white", stroke_fill="white", stroke_width=2
        )
    else:
        draw.text((x, y), kanji, font=font, fill="black")

    # Draw the kanji on the image
    draw.text((x, y), kanji, font=font, fill="black")
    # image = image.convert("1")
    return add_noise(image, blurr_factor)  # .convert("1")


import fontTools
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from itertools import chain


def get_chars(font_path):
    font = TTFont(font_path)
    return list(
        chain.from_iterable(
            [y + (Unicode[y[0]],) for y in x.cmap.items()] for x in font["cmap"].tables
        )
    )


supported_chars = {}
for font, _ in FONTS:
    print("Computing supported chars for %s." % font)
    font_path = "%s/%s.ttf" % (FONT_DIR, font)
    supported_chars[font_path] = get_chars(font_path)
    print("Computed supported chars for %s." % font)


@functools.lru_cache
def is_character_supported(font_path, character):
    font = TTFont(font_path)
    chars = supported_chars[font_path]
    code_points = {c[0] for c in chars}
    code_point = ord(character)  # search character
    print(Unicode[code_point])
    return code_point in code_points


# Example usage
for kanji_to_draw, number in kanji_dict.items():
    for (font, additional_scale) in FONTS:
        font_path = "%s/%s.ttf" % (FONT_DIR, font)
        if not is_character_supported(font_path, kanji_to_draw):
            continue
        for image_size in [20, 25, 30, 35]:
            for delta_x in [0]:
                for delta_y in [0]:
                    for scale_size in [1.0]:
                        for blurr_factor in [0.0, 0.5, 0.6]:
                            Path("%s/%s/" % (common.TRANING_DATA_DIR, number)).mkdir(
                                parents=True, exist_ok=True
                            )
                            output_image = create_kanji_image(
                                kanji_to_draw,
                                image_size=image_size,
                                scale_size=scale_size,
                                font_path=font_path,
                                delta_x=delta_x,
                                delta_y=delta_y,
                                adjustment_scale=additional_scale,
                                blurr_factor=blurr_factor,
                            )
                            file_name_prefix = (
                                "%s/%s/seq_%s_size%s_scale%s_dx%s_dy%s_font%s_addscale%s_blurr%s"
                                % (
                                    common.TRANING_DATA_DIR,
                                    number,
                                    number,
                                    image_size,
                                    scale_size,
                                    delta_x,
                                    delta_y,
                                    font,
                                    additional_scale,
                                    blurr_factor,
                                )
                            )
                            output_image.save("%s.png" % file_name_prefix, "PNG")
                            inverted = ImageOps.invert(output_image)
                            inverted.save("%s_inv.png" % file_name_prefix, "PNG")
                            if image_size != 30:
                                continue
                            for quality in [50, 5]:
                                jpg_file_name = "%s_compression%s.jpg" % (
                                    file_name_prefix,
                                    quality,
                                )
                                output_image.save(
                                    jpg_file_name, "JPEG", quality=quality
                                )
                                with Image.open(jpg_file_name) as compressed_img:
                                    compressed_img.save(
                                        "%s_compression%s.png"
                                        % (file_name_prefix, quality),
                                        "PNG",
                                    )
                                    inverted = ImageOps.invert(compressed_img)
                                    inverted.save(
                                        "%s_compression%s_inv.png"
                                        % (file_name_prefix, quality),
                                        "PNG",
                                    )
