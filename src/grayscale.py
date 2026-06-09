import cv2
import numpy as np

def grayscale(nome, escala=0.5):
    """
    Lê o vídeo, converte para tons de cinza, redimensiona, 
    salva o vídeo pré-processado e constrói a matriz M.
    """
    video_src = cv2.VideoCapture('./data/videoplayback.mkv')
    
    # Propriedades do vídeo original
    fps = video_src.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0 # Define um valor padrão caso o OpenCV não consiga ler o FPS
        
    orig_w = int(video_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(video_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Novas dimensões
    width = int(orig_w * escala)
    height = int(orig_h * escala)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    caminho_saida = f'./processed_data/{nome}.mp4'
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (width, height), isColor=False)
    frames = []
    shape = None
    
    while video_src.isOpened():
        ret, frame = video_src.read()
        if not ret:
            break
            
        # Converte para tons de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Redimensiona para reduzir custo computacional
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

        if shape is None:
            shape = resized.shape
            
        # Grava o frame processado no novo vídeo
        out.write(resized)
            
        # Vetoriza o frame (flattening) e adiciona à lista
        frames.append(resized.flatten())
        
    video_src.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Constrói a matriz M onde cada coluna é um frame vetorizado
    M = np.column_stack(frames)
    return M, shape