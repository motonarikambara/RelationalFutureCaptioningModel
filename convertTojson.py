import json
import csv


def main_origin():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './ponnet_data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train.json'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        for row in reader:
            # バグファイルを取り除く
            if video_id == 67:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                clip_id = str(video_id) + '_' + row[i] + '_' + row[i+1]
                sentence = row[i+2] + ' ' + row[i+3]
                json_list.append({'clip_id':clip_id, 'sentence':sentence})
                i += 4
            i = 1
            video_id += 1

            # sample-1kに対応
            if video_id >= 1000:
                break

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_para():
    print('********** JSONファイルを書き出す **********')

    json_list = dict()
    input_data = './ponnet_data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train_para.json'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        for row in reader:
            # バグファイルを取り除く
            if video_id == 67:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                clip_id = str(video_id) + '_' + row[i] + '_' + row[i+1]
                sentence = row[i+2] + ' ' + row[i+3]
                json_list[clip_id] = sentence
                i += 4
            i = 1
            video_id += 1

            # sample-1kに対応
            if video_id >= 1000:
                break

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_future():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './ponnet_data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train_future.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # バグファイルを取り除く
            if video_id == 67:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                sentence = row[i+2] + ' ' + row[i+3]
                sentence_list.append(sentence)
                i += 4
            i = 1

            while row[i] != '' and row[i+1] != '':
                clip_id = str(video_id) + '_' + row[i] + '_' + row[i+1]
                if k + 1 <= len(sentence_list) - 1:
                    json_list.append({'clip_id':clip_id, 'sentence':sentence_list[k+1]})
                    k += 1
                else:
                    break
                i += 4
            video_id += 1
            i = 1
            k = 0
            sentence_list = []

            # sample-1kに対応
            if video_id >= 1000:
                break


    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_future_para():
    print('********** JSONファイルを書き出す **********')

    json_list = dict()
    input_data = './ponnet_data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train_future_para.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # バグファイルを取り除く
            if video_id == 67:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                sentence = row[i+2] + ' ' + row[i+3]
                sentence_list.append(sentence)
                i += 4
            i = 1

            while row[i] != '' and row[i+1] != '':
                clip_id = str(video_id) + '_' + row[i] + '_' + row[i+1]
                if k + 1 <= len(sentence_list) - 1:
                    json_list[clip_id] = sentence_list[k+1]
                    k += 1
                else:
                    break
                i += 4
            i = 1
            video_id += 1
            k = 0
            sentence_list = []

            # sample-1kに対応
            if video_id >= 1000:
                break

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


if __name__ == '__main__':
    your_choice = input('origin : 0, para : 1, future : 2, future_para : 3\n')
    if your_choice == '0':
        main_origin()
    elif your_choice == '1':
        main_para()
    elif your_choice == '2':
        main_future()
    elif your_choice == '3':
        main_future_para()
