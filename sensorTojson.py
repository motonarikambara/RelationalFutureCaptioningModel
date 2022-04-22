import json
import csv

def main():
    print('********** JSONファイルを書き出す **********')

    json_list = []
    input_data = './BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = './annotations/BDD-X/bddx_sensor.json'
    sensor_dir = './BDD-X-Dataset/train/info/'

    # CSV ファイルの読み込み
    with open(input_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        video_id = 0
        filename_list = []
        for row in reader:
            # trainに対応
            if video_id < 1000:
                video_id += 1
                continue
            if row[0] == "":
                continue
            file_name = row[0].replace('https://s3-us-west-2.amazonaws.com/sq8geewpqu/train/', '')
            file_name = file_name.replace('.mov', "")
            filename_list.append(file_name)

    for filename in filename_list:
        json_name = sensor_dir + filename + ".json"
        dir_name = './BDD-X-Dataset/frames/'
        json_open = open(json_name, 'r')
        json_load = json.load(json_open)
        speed_post = 0
        for num in range(len(json_load['locations'])):
            num_zero = str(num).zfill(4)
            img_path = dir_name + num_zero + '.png'
            speed = json_load['locations'][num]['speed']
            course = json_load['locations'][num]['course']
            accel = speed - speed_post
            speed_post = speed
            json_list.append({'img_path':img_path, 'speed':speed, 'course':course, 'accelerater':accel})
        speed_post = 0

    # 辞書オブジェクトをJSONファイルへ出力
    with open(output_file, mode='wt', encoding='utf-8') as file:
        json.dump(json_list, file, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    main()
