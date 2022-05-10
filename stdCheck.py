import json
import numpy as np

def main():
    input_file = './annotations/BDD-X/bddx_sensor_train.json'

    json_open = open(input_file, 'r')
    json_load = json.load(json_open)

    speed_list = []
    acc_list = []
    crs_list = []
    crs_vel_list = []

    for v in json_load:
        speed_list.append(float(v['speed']))
        acc_list.append(float(v['accelerater']))
        crs_list.append(float(v['course']))
        crs_vel_list.append(float(v['course_vel']))
    speed_std = np.std(speed_list)
    acc_std = np.std(acc_list)
    crs_std = np.std(crs_list)
    crs_vel_std = np.std(crs_vel_list)

    print('std')
    print('speed :', speed_std)
    print('accelerater :', acc_std)
    print('crs :', crs_std)
    print('crs_vel :', crs_vel_std)


if __name__ == '__main__':
    main()
