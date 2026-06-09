import cv2
import numpy as np
import svd
import video_processing as vp
import plots

def run_numpy(k_max, escala=1):
    '''
    SVD com NumPy
    '''
    # Criação da matriz M
    M, shape = vp.grayscale('grayscale_resized', escala)

    U, S_diag, V_T = svd.numpy_svd(M)
    L, S_mask = svd.rank_k_approximation(M, U, S_diag, V_T, k_max)

    # Reconstrói os vídeos de background e foreground
    num_frames = M.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    out_bg = cv2.VideoWriter(f'processed_data/bg_numpy{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    out_mov = cv2.VideoWriter(f'processed_data/fg_numpy{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
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

    # Plotagem das métricas
    plots.plot_decaimento(S_diag, 'decaimento_numpy')
    plots.plot_variancia(S_diag, k_max, 'variancia_numpy')
    plots.plot_erro(M, U, S_diag, V_T, k_max, 'erro_numpy')

def run_manual(k_max, escala=1):
    '''
    SVD com Método de Jacobi implementado manualmente
    '''
    # Criação da matriz M
    M, shape = vp.grayscale('grayscale_resized', escala)

    U, S_diag, V_T = svd.svd_manual(M)
    L, S_mask = svd.rank_k_approximation(M, U, S_diag, V_T, k_max)

    # Reconstrói os vídeos de background e foreground
    num_frames = M.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    out_bg = cv2.VideoWriter(f'processed_data/bg_manual{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    out_mov = cv2.VideoWriter(f'processed_data/fg_manual{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
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

    # Plotagem das métricas
    plots.plot_decaimento(S_diag, 'decaimento_manual')
    plots.plot_variancia(S_diag, k_max, 'variancia_manual')
    plots.plot_erro(M, U, S_diag, V_T, k_max, 'erro_manual')

def main():
    '''
    Parâmetros
    '''
    k_max = 3
    escala = 1

    run_numpy(k_max, escala)
    run_manual(k_max, escala)

if '__main__' == __name__:
    main()