import matplotlib.pyplot as plt
import numpy as np

def plot_decaimento(S_diag, nome):
    '''
    Plotagem do decaimento dos valores singulares sigma_i
    '''
    plt.plot(S_diag[:], marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.title(r'Decaimento dos Valores Singulares $\sigma_i$ (Log)')
    plt.xlabel('Índice i')
    plt.ylabel(r'Valor Singular $\sigma_i$ (Log)')
    plt.grid(True)
    plt.savefig(f'processed_data/{nome}.png', bbox_inches='tight')
    plt.show()

def plot_variancia(S_diag, k_max, nome):
    '''
    Plotagem da variância acumulada V(k)
    '''
    variancia_total = np.sum(S_diag**2)
    variancia_acumulada = [np.sum(S_diag[:i]**2) / variancia_total for i in range(1, k_max + 1)]
    plt.plot(range(1, k_max + 1), variancia_acumulada, marker='s', color='orange')
    plt.title('Variância Acumulada $V(k)$')
    plt.xlabel('Posto k')
    plt.ylabel('V(k)')
    plt.grid(True)
    plt.savefig(f'processed_data/{nome}.png', bbox_inches='tight')
    plt.show()

def plot_erro(M, U, S_diag, V_T, k_max, nome):
    '''
    Plotagem do erro de reconstrução (Norma de Frobenius) ||M - L||_F
    '''
    erros = []
    M = M.astype(np.float32)
    for k in range(1, k_max + 1):
        L_k = U[:, :k] @ np.diag(S_diag[:k]) @ V_T[:k, :]
        erro = np.linalg.norm(M - L_k, ord='fro')
        erros.append(erro)
        del L_k
        
    plt.plot(range(1, k_max + 1), erros, marker='^', color='red')
    plt.title('Erro de Reconstrução $||M - L||_F$')
    plt.xlabel('Posto k')
    plt.ylabel('Erro')
    plt.grid(True)
    plt.savefig(f'processed_data/{nome}.png', bbox_inches='tight')
    plt.show()

