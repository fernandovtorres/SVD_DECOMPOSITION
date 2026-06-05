import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def grayscale(nome, escala=0.5):
    """
    Lê o vídeo, converte para tons de cinza, redimensiona, 
    salva o vídeo pré-processado e constrói a matriz M.
    """
    video_src = cv2.VideoCapture('./data/videoplayback')
    
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

def numpy_svd(M):
    """
    Calcula a decomposição SVD usando a biblioteca numpy.
    """
    start_time = time.time()
    # SVD reduzida
    U, S, V_T = np.linalg.svd(M, full_matrices=False)
    tempo_execucao = time.time() - start_time
    print(f"Tempo de execução SVD com NumPy: {tempo_execucao:.4f} segundos")
    
    return U, S, V_T

def rank_k_approximation(M, U, S, V_T, k=1, threshold=30):
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = V_T[:k, :]

    # Plano de fundo isolado
    L = Uk @ Sk @ Vtk
    # Garante que os valores de L estejam no intervalo válido de pixels [0, 255]
    L = np.clip(L, 0, 255)

    # Matriz de movimento esparsa
    S = np.abs(M - L)
    # Aplica um limiar (threshold) para criar a máscara de movimento
    S_mask = np.where(S > threshold, 255, 0).astype(np.uint8)

    return L, S_mask

def plot(M, U, S_diag, V_T, k_max):
    '''
    Plotagem das métricas:
    1. Decaimento dos valores singulares sigma_i
    2. Variância acumulada V(k)
    3. Erro de reconstrução
    '''

    plt.figure(figsize=(16, 5))

    '''
    1. Decaimento dos valores singulares sigma_i
    '''
    plt.subplot(1, 3, 1)
    plt.plot(S_diag[:], marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.title(r'Decaimento dos Valores Singulares $\sigma_i$ (Log)')
    plt.xlabel('Índice i')
    plt.ylabel(r'Valor Singular $\sigma_i$ (Log)')
    plt.grid(True)
    
    '''
    2. Cálculo da variância acumulada V(k)
    '''
    variancia_total = np.sum(S_diag**2)
    variancia_acumulada = [np.sum(S_diag[:i]**2) / variancia_total for i in range(1, k_max + 1)]
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, k_max + 1), variancia_acumulada, marker='s', color='orange')
    plt.title('Variância Acumulada $V(k)$')
    plt.xlabel('Posto k')
    plt.ylabel('V(k)')
    plt.grid(True)
    
    '''
    3. Erro de reconstrução (Norma de Frobenius) ||M - L||_F
    '''
    erros = []
    for k in range(1, k_max + 1):
        L_k = U[:, :k] @ np.diag(S_diag[:k]) @ V_T[:k, :]
        erro = np.linalg.norm(M - L_k, ord='fro')
        erros.append(erro)
        
    plt.subplot(1, 3, 3)
    plt.plot(range(1, k_max + 1), erros, marker='^', color='red')
    plt.title('Erro de Reconstrução $||M - L||_F$')
    plt.xlabel('Posto k')
    plt.ylabel('Erro')
    plt.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('processed_data/metricas.png', bbox_inches='tight')
    plt.show()

def main():
    '''
    Separação do background (L) e dos objetos em movimento (S_mask)
    '''
    k_max = 3
    escala = 0.5
    print(f'Escala: {escala}, k_max = {k_max}')

    M, shape = grayscale('grayscale_resized', escala)
    U, S_diag, V_T = numpy_svd(M)
    L, S_mask = rank_k_approximation(M, U, S_diag, V_T, k_max)

    num_frames = M.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    out_bg = cv2.VideoWriter(f'processed_data/fundo{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    out_mov = cv2.VideoWriter(f'processed_data/movimento{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    
    for i in range(num_frames):
        # Pega a coluna i do Fundo (L) e remodela para a imagem 2D
        frame_fundo = L[:, i].reshape(shape).astype(np.uint8)
        
        # Pega a coluna i da Máscara de Movimento (S_mask) e remodela
        frame_mask = S_mask[:, i].reshape(shape).astype(np.uint8)
        
        # Grava os frames em seus respectivos arquivos
        out_bg.write(frame_fundo)
        out_mov.write(frame_mask)
        
    out_bg.release()
    out_mov.release()

    plot(M, U, S_diag, V_T, k_max)

if '__main__' == __name__:
    main()