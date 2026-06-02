import cv2
import numpy as np

def manual_svd(M):
    M_T_M = M.T @ M
    eigenvalues, V = np.linalg.eigh(M_T_M)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    eigenvalues[eigenvalues < 0] = 0
    S = np.sqrt(eigenvalues)

    m, n = M.shape
    U = np.zeros((m, n))

    for i in range(n):
        if S[i] > 1e-9:
            U[:, i] = (M @ V[:, i]) / S[i]
    return U, S, V.T

def rank_k_approximation(U, S, Vt, k=1):
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    return Uk @ Sk @ Vtk

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
    
    U, S, Vt = manual_svd(M)

if '__main__' == __name__:
    main()

