import json
import re
import os
import csv

def main_train():
    csv_name = './annotations/ponnet/train.csv'
    output_file = './annotations/ponnet/captioning_train_para.json'
    with open(csv_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        clip_dict = {}
        for row in reader:
            clip_dict[row[0]] = row[1]

    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(clip_dict, file, ensure_ascii=False, indent=None)


def main_valid():
    csv_name = './annotations/ponnet/val.csv'
    output_file = './annotations/ponnet/captioning_val_para.json'
    with open(csv_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        clip_dict = {}
        for row in reader:
            clip_dict[row[0]] = row[1]

    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(clip_dict, file, ensure_ascii=False, indent=None)


def main_test():
    csv_name = './annotations/ponnet/test.csv'
    output_file = './annotations/ponnet/captioning_test_para.json'
    with open(csv_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        clip_dict = {}
        for row in reader:
            clip_dict[row[0]] = row[1]

    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(clip_dict, file, ensure_ascii=False, indent=None)


if __name__ == '__main__':
    your_choice = input('train : 0, valid : 1, test : 2\n')
    if your_choice == '0':
        main_train()
    elif your_choice == '1':
        main_valid()
    elif your_choice == '2':
        main_test()
