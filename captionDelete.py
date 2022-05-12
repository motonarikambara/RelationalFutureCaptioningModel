import json
import re
import os

def main():
    json_name = './annotations/BDD-X/captioning_train.json'
    output_file = './annotations/BDD-X/_captioning_train.json'
    json_open = open(json_name, 'r')
    json_load = json.load(json_open)
    cap_list = []

    for v in json_load:
        clip_id = v['clip_id']
        pkl_name = './out/pretrain/train/' + clip_id + '.pkl'
        is_file = os.path.isfile(pkl_name)
        if is_file:
            cap_list.append(v)
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(cap_list, file, ensure_ascii=False, indent=None)


if __name__ == '__main__':
    main()
