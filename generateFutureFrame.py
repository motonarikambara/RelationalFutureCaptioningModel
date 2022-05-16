import cv2
from os import makedirs
import os
from os.path import splitext, dirname, basename, join
import csv
import subprocess
from tqdm import tqdm


def save_frames(video_path: str, frame_dir: str,
                name="frame", ext="png", file_name="tmp"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    v_name = splitext(basename(video_path))[0]
    # if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
    #     frame_dir = dirname(frame_dir)
    # frame_dir_ = join(frame_dir, v_name)


    # makedirs(frame_dir_, exist_ok=True)
    # base_path = join(frame_dir_, name)

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                # cv2.imwrite("{}_{}.{}".format(base_path, "0", ext),
                #             frame)
                continue
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                # filled_second = str(second - 5)
                if second <= 4:
                    continue
                # if second > 6:
                #     break
                # cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                #             frame)
                if second == 5:
                    cv2.imwrite(os.path.join(frame_dir, "{}.{}".format(file_name, ext)), frame)
                    idx = 0
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                elif second > 6:
                    break
        else:
            break

if __name__ == "__main__":
    frame_dir = "./ponnet_data/future_frames/"
    clip_file = "./ponnet_data/1000samples.csv"

    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        # reader = csv.reader(f)
        i = 0
        for row in tqdm(reader):
            file_name = '_' + row[0] + '.mp4'
            # dir_name = frame_dir + file_name.replace(".mov", "")
            save_frames("./ponnet_data/samples/" + file_name, frame_dir, file_name=row[0])
