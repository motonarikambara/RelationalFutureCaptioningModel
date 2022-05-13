from re import T
import subprocess
import csv
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os

tsv_file = "./BDD-X-Dataset/BDD-X-Annotations_v1.csv"
clip_file = "./BDD-X-Dataset/BDD-X-Annotations_v1.csv"

def main_train():
    # ffmpegによる動画の切り出し
    id_list = []
    output_names = []
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        video_id = 0
        for row in reader:
            # sampleの1000データに対応
            if video_id < 1000 or video_id >= 10194:
                video_id += 1
                continue
            id = video_id
            id_list.append(int(id))
            video_id += 1


    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        # reader = csv.reader(f)
        i = 0
        j = 1
        for row in reader:
            # sampleの1000データに対応
            if i < 1000 or i >= 10194:
                i += 1
                continue
            if i in id_list:
                tmp_names = []
                while row[j] != '' and row[j+1] != '' and row[j+4] != '' and row[j+5] != '':
                    output_file_name_1 = str(i) + "_" + row[j] + "_" + row[j+1] + ".pkl"
                    output_file_name_2 = str(i) + "_" + row[j+4] + "_" + row[j+5] + ".pkl"
                    tmp_names.append(output_file_name_1)
                    tmp_names.append(output_file_name_2)
                    output_names.append(tmp_names)
                    j += 4
                    tmp_names = []
            i += 1
            j = 1

    tmp_dir = './out/pretrain/tmp/'
    output_dir = './out/pretrain/future_train/'
    src_dir = './out/pretrain/train/'

    for pair_name in tqdm(output_names):
        src_name = src_dir + pair_name[0]
        src_out_name = src_dir + pair_name[1]
        is_flie_1 = os.path.isfile(src_name)
        is_file_2 = os.path.isfile(src_out_name)
        if (not is_flie_1) or (not is_file_2):
            continue
        tmp_in_name = tmp_dir + pair_name[0]
        tmp_out_name = tmp_dir + pair_name[1]
        # output_name = output_dir + pair_name[1]

        subprocess.run('cp ' + src_name + ' ' + tmp_dir, shell=True)
        # subprocess.run('cd ' + tmp_dir, shell=True)
        subprocess.run('mv ' + tmp_in_name + ' ' + tmp_out_name, shell=True)
        subprocess.run('mv ' + tmp_out_name + ' ' + output_dir, shell=True)
        # subprocess.run('cd ./../../../', shell=True)


def main_valid():
    # ffmpegによる動画の切り出し
    id_list = []
    output_names = []
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        video_id = 0
        for row in reader:
            # sampleの1000データに対応
            if video_id < 10194 or video_id >= 11596:
                video_id += 1
                continue
            id = video_id
            id_list.append(int(id))
            video_id += 1


    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        # reader = csv.reader(f)
        i = 0
        j = 1
        for row in reader:
            # sampleの1000データに対応
            if i < 10194 or i >= 11596:
                i += 1
                continue
            if i in id_list:
                tmp_names = []
                while row[j] != '' and row[j+1] != '' and row[j+4] != '' and row[j+5] != '':
                    output_file_name_1 = str(i) + "_" + row[j] + "_" + row[j+1] + ".pkl"
                    output_file_name_2 = str(i) + "_" + row[j+4] + "_" + row[j+5] + ".pkl"
                    tmp_names.append(output_file_name_1)
                    tmp_names.append(output_file_name_2)
                    output_names.append(tmp_names)
                    j += 4
                    tmp_names = []
            i += 1
            j = 1

    tmp_dir = './out/pretrain/tmp/'
    output_dir = './out/pretrain/future_valid/'
    src_dir = './out/pretrain/valid/'

    for pair_name in tqdm(output_names):
        src_name = src_dir + pair_name[0]
        src_out_name = src_dir + pair_name[1]
        is_flie_1 = os.path.isfile(src_name)
        is_file_2 = os.path.isfile(src_out_name)
        if (not is_flie_1) or (not is_file_2):
            continue
        tmp_in_name = tmp_dir + pair_name[0]
        tmp_out_name = tmp_dir + pair_name[1]
        # output_name = output_dir + pair_name[1]

        subprocess.run('cp ' + src_name + ' ' + tmp_dir, shell=True)
        # subprocess.run('cd ' + tmp_dir, shell=True)
        subprocess.run('mv ' + tmp_in_name + ' ' + tmp_out_name, shell=True)
        subprocess.run('mv ' + tmp_out_name + ' ' + output_dir, shell=True)
        # subprocess.run('cd ./../../../', shell=True)


def main_test():
    # ffmpegによる動画の切り出し
    id_list = []
    output_names = []
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        video_id = 0
        for row in reader:
            # sampleの1000データに対応
            if video_id < 11596:
                video_id += 1
                continue
            id = video_id
            id_list.append(int(id))
            video_id += 1


    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        # reader = csv.reader(f)
        i = 0
        j = 1
        for row in reader:
            # sampleの1000データに対応
            if i < 11596:
                i += 1
                continue
            if i in id_list:
                tmp_names = []
                while row[j] != '' and row[j+1] != '' and row[j+4] != '' and row[j+5] != '':
                    output_file_name_1 = str(i) + "_" + row[j] + "_" + row[j+1] + ".pkl"
                    output_file_name_2 = str(i) + "_" + row[j+4] + "_" + row[j+5] + ".pkl"
                    tmp_names.append(output_file_name_1)
                    tmp_names.append(output_file_name_2)
                    output_names.append(tmp_names)
                    j += 4
                    tmp_names = []
            i += 1
            j = 1

    tmp_dir = './out/pretrain/tmp/'
    output_dir = './out/pretrain/future_test/'
    src_dir = './out/pretrain/test/'

    for pair_name in tqdm(output_names):
        src_name = src_dir + pair_name[0]
        src_out_name = src_dir + pair_name[1]
        is_flie_1 = os.path.isfile(src_name)
        is_file_2 = os.path.isfile(src_out_name)
        if (not is_flie_1) or (not is_file_2):
            continue
        tmp_in_name = tmp_dir + pair_name[0]
        tmp_out_name = tmp_dir + pair_name[1]
        # output_name = output_dir + pair_name[1]

        subprocess.run('cp ' + src_name + ' ' + tmp_dir, shell=True)
        # subprocess.run('cd ' + tmp_dir, shell=True)
        subprocess.run('mv ' + tmp_in_name + ' ' + tmp_out_name, shell=True)
        subprocess.run('mv ' + tmp_out_name + ' ' + output_dir, shell=True)
        # subprocess.run('cd ./../../../', shell=True)


if __name__ == '__main__':
    your_choice = input('train : 0, valid : 1, test : 2\n')
    if your_choice == '0':
        main_train()
    elif your_choice == '1':
        main_valid()
    elif your_choice == '2':
        main_test()
