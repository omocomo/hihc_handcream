import numpy as np
import cv2
import json
import os

pose_point_pair = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 9], [9, 8], [8, 10], [10, 5]]

face_point_pair = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], 
    [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], 
    [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [28, 29], 
    [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], 
    [38, 39], [39, 40], [40, 41], [36, 41], [42, 43], [43, 44], [44, 45], [45, 46], 
    [46, 47], [42, 47], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], 
    [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [48, 59], [60, 61], [61, 62], [62, 63], [63, 64]
    ]

hand_point_pair = [[i, i+1] for i in range(0, 20)]

def create_stick(file, keypoints, save_path):
    for keypoint in range(len(keypoints)):
        pose = keypoints[keypoint][:22]
        face = keypoints[keypoint][30:170]
        left_hand = keypoints[keypoint][170:212]
        right_hand = keypoints[keypoint][212:]

        part = [pose, face, left_hand, right_hand]
        part_num_points = [11, 68, 21, 21]
        part_pair = [pose_point_pair, face_point_pair, hand_point_pair, hand_point_pair]

        # create paper
        img = np.zeros((1500, 1500), np.uint8)+255

        for p in range(len(part)):
            x = part[p][0::2]
            y = part[p][1::2]

            # draw points
            for i in range(part_num_points[p]):
                cv2.circle(img, (int(x[i]*2048), int(y[i]*1152)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  
            
            # draw lines
            for pair in part_pair[p]:
                cv2.line(img, (int(x[pair[0]]*2048), int(y[pair[0]]*1152)), (int(x[pair[1]]*2048), int(y[pair[1]]*1152)), (0, 0, 255), 2)

        cv2.imwrite(save_path + f'/{file[:-5]}_{keypoint:03}.jpg', img)


if __name__ == '__main__':

    path = './201215/'

    for file in os.listdir(path):
        if 'json' in file:
            f = open(path + file, encoding="UTF-8")
            keypoints = json.loads(f.read())

            save_path = path + file[:-5]
            os.mkdir(save_path)
            
            create_stick(file, keypoints, save_path)
    