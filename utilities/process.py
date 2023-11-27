import os
import cv2

root_path = 'C:/Users/Administrator.DESKTOP-ARR9GVR/Desktop/files'
for root, dirs, files in os.walk(root_path):
    for file in files:
        # 获取文件相对于当前文件夹的路径
        file_path = os.path.join(root, file)
        img = cv2.imread(file_path)
        if '23l.jpg' == file:
            l = img[:, 100:356, :]
        elif '36l.jpg' == file:
            l = img[:, 200:456, :]
        elif '52l.jpg' == file:
            l = img[:, :256, :]
        elif '72l.jpg' == file:
            l = img[:, 100:356, :]
        elif '105l.jpg' == file:
            l = img[:, 100:356, :]
        elif '196l.jpg' == file:
            l = img[:, 200:456, :]
        elif '445l.jpg' == file:
            l = img[:, 25:281, :]
        else:
            continue
        cv2.imwrite(file_path.replace('.jpg', 'r.jpg'), l)
