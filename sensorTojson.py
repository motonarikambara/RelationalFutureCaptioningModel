import json
import csv
import os

def main_train():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_sensor_train.json'
    sensor_dir = './BDD-X-Dataset/train/info/'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_list = []
        for row in reader:
            # trainに対応
            if video_id < 1000 or video_id >= 10194:
                video_id += 1
                continue
            if row[0] == "":
                video_id += 1
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_list.append(file_name)
            video_id += 1

    for filename in filename_list:
        json_name = sensor_dir + filename + ".json"
        dir_name = './BDD-X-Dataset/frames/' + filename + '/'
        json_open = open(json_name, 'r')
        json_load = json.load(json_open)
        for num in range(len(json_load['locations'])):
            num_zero = str(num).zfill(4)
            img_path = dir_name + 'frame_' + num_zero + '.png'
            is_file = os.path.isfile(img_path)
            if not is_file:
                continue
            speed = json_load['locations'][num]['speed']
            course = json_load['locations'][num]['course']
            if num == 0:
                speed_post = speed
                course_post = course
            accel = speed - speed_post
            course_vel = course - course_post
            if -270 < course_vel and  course_vel <= -90:
                course_vel = -90
            elif course_vel <= -270:
                course_vel += 360
            elif 90 <= course_vel and  course_vel < 270:
                course_vel = 90
            elif 270 <= course_vel:
                course_vel -= 360
            speed_post = speed
            course_post = course
            if course == -1:
                course = 0
            else:
                course -= 180
            json_list.append({'img_path':img_path, 'speed':speed, 'course':course, 'accelerater':accel, 'course_vel':course_vel})

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_valid():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_sensor_valid.json'
    sensor_dir = './BDD-X-Dataset/train/info/'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_list = []
        for row in reader:
            # valに対応
            if video_id < 10194 or video_id >= 11596:
                video_id += 1
                continue
            if row[0] == "":
                video_id += 1
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_list.append(file_name)
            video_id += 1

    for filename in filename_list:
        json_name = sensor_dir + filename + ".json"
        dir_name = './BDD-X-Dataset/frames/' + filename + '/'
        json_open = open(json_name, 'r')
        json_load = json.load(json_open)
        for num in range(len(json_load['locations'])):
            num_zero = str(num).zfill(4)
            img_path = dir_name + 'frame_' + num_zero + '.png'
            is_file = os.path.isfile(img_path)
            if not is_file:
                continue
            speed = json_load['locations'][num]['speed']
            course = json_load['locations'][num]['course']
            if num == 0:
                speed_post = speed
                course_post = course
            accel = speed - speed_post
            course_vel = course - course_post
            if -270 < course_vel and  course_vel <= -90:
                course_vel = -90
            elif course_vel <= -270:
                course_vel += 360
            elif 90 <= course_vel and  course_vel < 270:
                course_vel = 90
            elif 270 <= course_vel:
                course_vel -= 360
            speed_post = speed
            course_post = course
            json_list.append({'img_path':img_path, 'speed':speed, 'course':course, 'accelerater':accel, 'course_vel':course_vel})

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


def main_test():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_sensor_test.json'
    sensor_dir = './BDD-X-Dataset/train/info/'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_list = []
        for row in reader:
            # valに対応
            if video_id < 11596:
                video_id += 1
                continue
            if row[0] == "":
                video_id += 1
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_list.append(file_name)
            video_id += 1

    for filename in filename_list:
        json_name = sensor_dir + filename + ".json"
        dir_name = './BDD-X-Dataset/frames/' + filename + '/'
        json_open = open(json_name, 'r')
        json_load = json.load(json_open)
        for num in range(len(json_load['locations'])):
            num_zero = str(num).zfill(4)
            img_path = dir_name + 'frame_' + num_zero + '.png'
            is_file = os.path.isfile(img_path)
            if not is_file:
                continue
            speed = json_load['locations'][num]['speed']
            course = json_load['locations'][num]['course']
            if num == 0:
                speed_post = speed
                course_post = course
            accel = speed - speed_post
            course_vel = course - course_post
            if -270 < course_vel and  course_vel <= -90:
                course_vel = -90
            elif course_vel <= -270:
                course_vel += 360
            elif 90 <= course_vel and  course_vel < 270:
                course_vel = 90
            elif 270 <= course_vel:
                course_vel -= 360
            speed_post = speed
            course_post = course
            json_list.append({'img_path':img_path, 'speed':speed, 'course':course, 'accelerater':accel, 'course_vel':course_vel})

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
