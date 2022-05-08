import json
import csv
import os

def main_train():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_id_train.json'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_dict = {}
        for row in reader:
            # trainに対応
            if video_id < 1000 or video_id >= 10194:
                video_id += 1
                continue
            if row[0] == "":
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_dict = {}
            filename_dict['clip_id'] = file_name
            filename_dict['video_id'] = video_id
            video_id += 1
            json_list.append(filename_dict)

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_valid():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_id_valid.json'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_dict = {}
        for row in reader:
            # trainに対応
            if video_id < 10194 or video_id >= 11596:
                video_id += 1
                continue
            if row[0] == "":
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_dict = {}
            filename_dict['clip_id'] = file_name
            filename_dict['video_id'] = video_id
            video_id += 1
            json_list.append(filename_dict)

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_test():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_id_test.json'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_dict = {}
        for row in reader:
            # trainに対応
            if video_id < 11596:
                video_id += 1
                continue
            if row[0] == "":
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_dict = {}
            filename_dict['clip_id'] = file_name
            filename_dict['video_id'] = video_id
            video_id += 1
            json_list.append(filename_dict)

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    your_choice = input('train : 0, valid : 1, test : 2\n')
    if your_choice == '0':
        main_train()
    elif your_choice == '1':
        main_valid()
    elif your_choice == '2':
        main_test()
