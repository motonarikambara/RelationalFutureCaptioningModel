import json
import csv

def main_test():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_test.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # testに対応
            if video_id < 11596:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                sentence = row[i+2] + ' ' + row[i+3]
                sentence = sentence.lower()
                sentence = sentence.replace('.', '')
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_test_para():
    print('********** JSONファイルを書き出す **********')

    json_list = dict()
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_test_para.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # testに対応
            if video_id < 11596:
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_val():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_val.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # valに対応
            if video_id < 10194 or video_id >= 11596:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                sentence = row[i+2] + ' ' + row[i+3]
                sentence = sentence.lower()
                sentence = sentence.replace('.', '')
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_val_para():
    print('********** JSONファイルを書き出す **********')

    json_list = dict()
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_val_para.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # valに対応
            if video_id < 10194 or video_id >= 11596:
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_train():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # trainに対応
            if video_id < 1000 or video_id >= 10194:
                video_id += 1
                continue

            while row[i] != '' and row[i+1] != '':
                sentence = row[i+2] + ' ' + row[i+3]
                sentence = sentence.lower()
                sentence = sentence.replace('.', '')
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_train_para():
    print('********** JSONファイルを書き出す **********')

    json_list = dict()
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/captioning_train_para.json'
    sentence_list = []

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        i = 1
        video_id = 0
        k = 0
        for row in reader:
            # trainに対応
            if video_id < 1000 or video_id >= 10194:
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

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


if __name__ == '__main__':
    your_choice = input('test : 0, test-para : 1, train : 2, train-para : 3, val : 4, val-para : 5,\n')
    if your_choice == '0':
        main_test()
    elif your_choice == '1':
        main_test_para()
    elif your_choice == '2':
        main_train()
    elif your_choice == '3':
        main_train_para()
    elif your_choice == '4':
        main_val()
    elif your_choice == '5':
        main_val_para()
