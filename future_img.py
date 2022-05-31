import cv2
import os
import csv
from tqdm import tqdm
import os
import numpy as np


def save_frame_sec(video_path, result_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = 4.0
    idx = 0
    for i in range(5):
        time = start_sec + i * 0.5
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * time))
        ret, frame = cap.read()
        im_name = "frames_{}.png".format(str(idx))
        save_path = os.path.join(result_path, im_name)
        idx += 1
        if ret:
            cv2.imwrite(save_path, frame)


def save_frames(video_path: str, frame_dir: str,
                name="frame", ext="png", file_name="tmp"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    save_frame_sec(video_path, frame_dir)

if __name__ == "__main__":
    frame_dir = "./ponnet_data/future_frames/"
    clip_file = "./ponnet_data/1000samples.csv"

    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        # reader = csv.reader(f)
        i = 0
        for row in tqdm(reader):
            file_name = row[0] + '.mp4'
            each_video = os.path.join(frame_dir, "_" + row[0])
            os.makedirs(each_video, exist_ok=True)
            # dir_name = frame_dir + file_name.replace(".mov", "")
            save_frames("./ponnet_data/samples/" + file_name, each_video, file_name=row[0])