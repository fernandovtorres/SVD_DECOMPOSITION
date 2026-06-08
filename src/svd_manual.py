import numpy as np
import time
import cv2
from svd import grayscale, rank_k_approximation, plot

def jacobi_eigen(A, tol=1e-8, max_iter=1000):
    """
    Calcula autovalores e autovetores de uma matriz simétrica real A
    usando o Método de Jacobi clássico.
    """
    n = A.shape[0]
    autovetores = np.eye(n)
    diag = A.astype(np.float64)
    
    for _ in range(max_iter):
        # Encontra o elemento de maior módulo fora da diagonal principal
        max_val = 0.0
        p, q = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(diag[i, j]) > max_val:
                    max_val = abs(diag[i, j])
                    p, q = i, j
                    
        # Condição de parada: se o maior valor fora da diagonal for menor que a tolerância
        if max_val < tol:
            break
            
        # Calcula o ângulo de rotação (theta)
        if diag[p, p] == diag[q, q]:
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan(2.0 * diag[p, q] / (diag[p, p] - diag[q, q]))
            
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Aplica a rotação de Givens em D e V
        D_pp = diag[p, p]
        D_qq = diag[q, q]
        D_pq = diag[p, q]
        
        # Atualiza os elementos da diagonal p e q
        diag[p, p] = c**2 * D_pp + s**2 * D_qq + 2 * c * s * D_pq
        diag[q, q] = s**2 * D_pp + c**2 * D_qq - 2 * c * s * D_pq
        diag[p, q] = diag[q, p] = 0.0
        
        # Atualiza o resto das linhas e colunas p e q
        for i in range(n):
            if i != p and i != q:
                D_ip = diag[i, p]
                D_iq = diag[i, q]
                diag[i, p] = diag[p, i] = c * D_ip + s * D_iq
                diag[i, q] = diag[q, i] = -s * D_ip + c * D_iq
                
        # Atualiza a matriz de autovetores
        for i in range(n):
            V_ip = autovetores[i, p]
            V_iq = autovetores[i, q]
            autovetores[i, p] = c * V_ip + s * V_iq
            autovetores[i, q] = -s * V_ip + c * V_iq
            
    # Os autovalores estão na matriz diagonal
    autovalores = np.diag(diag)
    return autovalores, autovetores

def svd_manual(M, tol=1e-8):
    """
    Calcula a Decomposição SVD utilizando o Método de Jacobi em M^T * M.
    """
    start_time = time.time()
    M_float = M.astype(np.float64)
    MtM = M_float.T @ M_float
    
    # Calcula autovalores e autovetores usando Jacobi
    autovalores, autovetores = jacobi_eigen(MtM, tol=tol)
    
    # Ordena os resultados em ordem decrescente
    idx_ordenado = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx_ordenado]
    autovetores = autovetores[:, idx_ordenado]
    
    # Calcula os valores singulares
    autovalores = np.maximum(autovalores, 0) # Evita raízes negativas por precisão de float
    S = np.sqrt(autovalores)
    
    # Filtra valores singulares não-nulos
    idx_nao_nulos = S > tol
    Sr = S[idx_nao_nulos]
    Vr = autovetores[:, idx_nao_nulos]
    
    # Calcula a matriz U (u_i = M * v_i / sigma_i)
    Ur = np.zeros((M.shape[0], len(Sr)))
    for i in range(len(Sr)):
        Ur[:, i] = (M @ Vr[:, i]) / Sr[i]
        
    tempo_execucao = time.time() - start_time
    print(f"Tempo de execução SVD com Método de Jacobi: {tempo_execucao:.4f} segundos")

    return Ur, Sr, Vr.T

def main():
    k_max = 3
    escala = 1
    print(f'Escala: {escala}, k_max = {k_max}')    

    M, shape = grayscale('grayscale_resized', escala)
    Ur, Sr, Vr_T = svd_manual(M)
    L, S_mask = rank_k_approximation(M, Ur, Sr, Vr_T, k_max)

    num_frames = M.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    out_bg = cv2.VideoWriter(f'processed_data/fundo_manual{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    out_mov = cv2.VideoWriter(f'processed_data/movimento_manual{k_max}.mp4', fourcc, 20.0, (shape[1], shape[0]), isColor=False)
    
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

    plot(M, Ur, Sr, Vr_T, k_max, f'metricas_manual{k_max}')

if '__main__' == __name__:
    main()