import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import plots

def rank_k_approximation(M, U, S, V_T, k=1, threshold=30):
    '''
    Aproxima a matriz de baixo posto (plano de fundo) e a matriz esparsa (objetos em movimento)
    '''
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = V_T[:k, :]

    # Plano de fundo isolado dentro do intervalo válido para a escala de cinzas
    L = np.clip(Uk @ Sk @ Vtk, 0, 255).astype(np.uint8)

    # Matriz de movimento esparsa
    S_diff = cv2.absdiff(M, L)
    # Cria a máscara de movimento
    S_mask = np.where(S_diff > threshold, 255, 0).astype(np.uint8)

    return L, S_mask

'''
SVD com a biblioteca NumPy
'''

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

'''
SVD manual com Método de Jacobi
'''

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
            
        # Calcula a variável auxiliar tau (cotangente de 2*theta)
        tau = (diag[q, q] - diag[p, p]) / (2.0 * diag[p, q])
        
        # Calcula a tangente (t) do ângulo
        if tau == 0.0:
            t = 1.0
        else:
            t = -np.sign(tau) / (np.abs(tau) + np.sqrt(1.0 + tau**2))
            
        # Calcula o cosseno (c) e o seno (s) a partir de t
        c = 1.0 / np.sqrt(1.0 + t**2)
        s = t * c
        
        # Aplica a rotação de Jacobi em D e V
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
    M_float = M.astype(np.float32)
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