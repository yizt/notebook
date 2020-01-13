# -*- coding: utf-8 -*-
"""
 @File    : font.py
 @Time    : 2020/1/7 上午9:07
 @Author  : yizuotian
 @Description    :
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


def to_unicode(glyph):
    return json.loads(f'"{glyph}"')


def get_font_chars(font_path):
    font = TTFont(font_path, fontNumber=0)
    glyph_names = font.getGlyphNames()
    char_list = []
    for idx, glyph in enumerate(glyph_names):
        if glyph[0] == '.':  # 跳过'.notdef', '.null'
            continue
        if glyph == 'union':
            continue
        if glyph[:3] == 'uni':
            glyph = glyph.replace('uni', '\\u')
        if glyph[:2] == 'uF':
            glyph = glyph.replace('uF', '\\u')
        if glyph == '\\uversal':
            continue

        char = to_unicode(glyph)
        char_list.append(char)
    return char_list


def is_char_visible(font, char):
    """
    是否可见字符
    :param font:
    :param char:
    :return:
    """
    gray = Image.fromarray(np.zeros((20, 20), dtype=np.uint8))
    draw = ImageDraw.Draw(gray)
    draw.text((0, 0), char, 100, font=font)
    visible = np.max(np.array(gray)) > 0
    return visible


def main():
    pass


def test_one_font(font_path):
    char_list = get_font_chars(font_path)
    single_char_list = list(set([c.strip() for c in char_list if len(c) == 1]))
    single_char_list.sort()
    font = ImageFont.truetype(font_path, size=10)
    visible_char_list = [c for c in single_char_list if is_char_visible(font, c)]
    invisible_char_list = [c for c in single_char_list if not is_char_visible(font, c)]
    print("font_path:{},char_list len:{},"
          "single_char_list len:{},visible_char_list len:{},"
          "visible_char_list len:{}".format(os.path.basename(font_path),
                                            len(char_list),
                                            len(single_char_list),
                                            len(visible_char_list),
                                            len(invisible_char_list)))
    print(char_list)
    print(''.join(single_char_list))
    print(''.join(visible_char_list))
    print(''.join(invisible_char_list))
    return visible_char_list


def test():
    font_dir = os.path.join(os.path.dirname(__file__), '../data/font')
    font_path_list = [os.path.join(font_dir, file_name) for file_name in os.listdir(font_dir)]
    print(font_path_list)
    char_list = []
    for p in font_path_list:
        char_list.extend(test_one_font(p))
    char_list = list(set(char_list))
    char_list.sort()
    import codecs
    with codecs.open('all_words.txt', 'w', 'utf-8') as f:
        for c in ''.join(char_list):
            f.write("{}\n".format(c))


def test_words():
    import codecs
    with codecs.open('all_words.txt', 'r') as f:
        lines = f.readlines()
    char_list = [l.strip() for l in lines]
    print(len(char_list), len(set(char_list)))


if __name__ == '__main__':
    test()
    test_words()
    # main()
