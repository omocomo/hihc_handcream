import os
import cv2

if __name__ == '__main__':
    path = './201215/'

    for file in os.listdir(path):
        if os.path.isdir(path + file):
            path_tmp = path + file + '/'

            # load stick img
            images = [img for img in os.listdir(path_tmp)]
            images.sort()

            # image to video
            fps = 30

            frame_array = []
            for i in range(len(images)):
                filename = path_tmp + images[i]

                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)

                frame_array.append(img)

            out = cv2.VideoWriter(path_tmp+'1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for i in range(len(frame_array)):
                out.write(frame_array[i])

            out.release()
