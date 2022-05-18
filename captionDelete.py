import json
import re
import os

def main_train():
    json_name = './annotations/ponnet/captioning_train.json'
    output_file = './annotations/ponnet/_captioning_train.json'
    json_open = open(json_name, 'r')
    json_load = json.load(json_open)
    cap_list = []

    for v in json_load:
        clip_id = v['clip_id']
        png_name = './ponnet_data/future_frames/' + clip_id + '.png'
        is_file = os.path.isfile(png_name)
        if is_file:
            cap_list.append(v)
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(cap_list, file, ensure_ascii=False, indent=None)


def main_valid():
    json_name = './annotations/ponnet/captioning_val.json'
    output_file = './annotations/ponnet/_captioning_valid.json'
    json_open = open(json_name, 'r')
    json_load = json.load(json_open)
    cap_list = []

    for v in json_load:
        clip_id = v['clip_id']
        png_name = './ponnet_data/future_frames/' + clip_id + '.png'
        is_file = os.path.isfile(png_name)
        if is_file:
            cap_list.append(v)
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(cap_list, file, ensure_ascii=False, indent=None)


def main_test():
    json_name = './annotations/ponnet/captioning_test.json'
    output_file = './annotations/ponnet/_captioning_test.json'
    json_open = open(json_name, 'r')
    json_load = json.load(json_open)
    cap_list = []

    for v in json_load:
        clip_id = v['clip_id']
        png_name = './ponnet_data/future_frames/' + clip_id + '.png'
        is_file = os.path.isfile(png_name)
        if is_file:
            cap_list.append(v)
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(cap_list, file, ensure_ascii=False, indent=None)


if __name__ == '__main__':
    your_choice = input('train : 0, valid : 1, test : 2\n')
    if your_choice == '0':
        main_train()
    elif your_choice == '1':
        main_valid()
    elif your_choice == '2':
        main_test()
