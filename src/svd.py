import cv2
import numpy as np

def main():
    video_source = cv2.VideoCapture('./processed_data/grayscale.mp4')
    frames = []

    pos_frame = video_source.get(1)

    while True:
        frame_r, frame = video_source.read()

        if frame_r:

            pos_frame = video_source.get(1)

            flat_frame = frame.flatten()
            frames.append(flat_frame)

        else:
            video_source.set(1, pos_frame-1)
            cv2.waitKey(1000)


        if cv2.waitKey(10) == 27:
            break
        if video_source.get(1) == video_source.get(7):
            break


    cv2.destroyAllWindows()
    video_source.release()

    M = np.array(frames).astype(float)
    print(M)
    print(M.shape)

if '__main__' == __name__:
    main()

