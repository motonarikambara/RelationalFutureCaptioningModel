import json

def main():
    input_file = './annotations/BDD-X/bddx_sensor_train.json'

    json_open = open(input_file, 'r')
    json_load = json.load(json_open)

    vel_max = 0.0
    vel_min = 10000.0
    vel_mean = 0.0
    acc_max = -100.0
    acc_min = 10000.0
    acc_mean = 0.0
    crs_max = -10000.0
    crs_min = 10000.0
    crs_mean = 0.0
    crs_vel_max = -1000.0
    crs_vel_min = 10000.0
    crs_vel_mean = 0.0
    count = 0
    for v in json_load:
        if float(v["speed"]) > vel_max:
            vel_max = v["speed"]
        if float(v["speed"]) < vel_min:
            vel_min = v["speed"]
        if float(v["accelerater"]) > acc_max:
            acc_max = v["accelerater"]
        if float(v["accelerater"]) < acc_min:
            acc_min = v["accelerater"]
        if float(v["course"]) > crs_max:
            crs_max = v["course"]
        if float(v["course"]) < crs_min:
            crs_min = v["course"]
        if float(v["course_vel"]) > crs_vel_max:
            crs_vel_max = v["course_vel"]
        if float(v["course_vel"]) < crs_vel_min:
            crs_vel_min = v["course_vel"]
        vel_mean += float(v["speed"])
        acc_mean += float(v["accelerater"])
        crs_mean += float(v["course"])
        crs_vel_mean += float(v["course_vel"])

        count += 1

    print("vel_min:", vel_min, "vel_max:", vel_max, "vel_mean:", vel_mean / count)
    print("acc_min:", acc_min, "acc_max:", acc_max, "acc_mean:", acc_mean / count)
    print("crs_min:", crs_min, "crs_max:", crs_max, "crs_mean:", crs_mean / count)
    print("crs_vel_min:", crs_vel_min, "crs_vel_max:", crs_vel_max, "crs_vel_mean:", crs_vel_mean / count)



if __name__ == '__main__':
    main()
