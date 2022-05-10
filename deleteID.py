import json
import re

def main():
    json_name = './annotations/BDD-X/bddx_sensor_train.json'
    json_open = open(json_name, 'r')
    json_load = json.load(json_open)
    zero_list = []

    for v in json_load:
        if v['course'] == 0:
            zero_file = re.sub('/frame_.*', '', v['img_path']) + '\n'
            zero_list.append(zero_file)

    f = open('./annotations/BDD-X/zero_json.txt', 'w')
    f.writelines(zero_list)
    f.close()


if __name__ == '__main__':
    main()
