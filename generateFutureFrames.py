import cv2
from os import makedirs
from os.path import splitext, dirname, basename, join
import csv
import subprocess
from tqdm import tqdm
import os
import numpy as np
from PIL import Image



def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2 + 80,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2 + 80))


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def save_frames(video_path: str, frame_dir: str,
                name="frame", ext="png", file_name="tmp"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    # v_name = splitext(basename(video_path))[0]
    # if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
    #     frame_dir = dirname(frame_dir)
    # frame_dir_ = join(frame_dir, v_name)


    v_name = splitext(basename(video_path))[0]
    # if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
    #     frame_dir = dirname(frame_dir)

    # makedirs(frame_dir, exist_ok=True)
    # base_path = join(frame_dir, name)
    base_path = frame_dir

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                # cv2.imwrite("{}_{}.{}".format(base_path, "0", ext),
                #             frame)
                pass
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                # filled_second = str(second)
                if second > 1:
                    break
                if second == 1:
                    # cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                    #             frame)
                    # 切り抜き
                    frame = cv2.resize(frame, dsize=(500, 500))
                    frame = cv2pil(frame)
                    frame = crop_center(frame, 224, 224)
                    frame = pil2cv(frame)
                    frame = cv2.resize(frame, dsize=(64, 64))
                    cv2.imwrite(os.path.join(base_path, "{}.{}".format(file_name, ext)), frame)
                idx = 0
        else:
            break

if __name__ == "__main__":
    frame_dir = "./ponnet_data/64_center_future_frames/"
    clip_file = "./ponnet_data/1000samples.csv"

    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        # reader = csv.reader(f)
        i = 0
        for row in tqdm(reader):
            file_name = '_' + row[0] + '.mp4'
            # dir_name = frame_dir + file_name.replace(".mov", "")
            save_frames("./ponnet_data/samples/" + file_name, frame_dir, file_name=row[0])