# Decomposição em Valores Singulares (SVD) aplicada a Vídeos

## Sobre o Projeto

Este projeto tem como objetivo demonstrar o funcionamento da **Decomposição em Valores Singulares (SVD)** através da sua aplicação em vídeos curtos, visando separar os objetos em movimento (primeiro plano) do cenário estático (fundo).

A hipótese central baseia-se na premissa de que um vídeo gravado por uma câmera estática possui um plano de fundo altamente correlacionado ao longo do tempo. Devido a essa forte correlação temporal, o fundo pode ser representado por uma matriz de baixo posto. Desse modo, o vídeo original pode ser modelado matematicamente como a soma de dois componentes matriciais:

$$M = L + S$$

Onde:

* **$L$ (Low-rank)**: Representa a aproximação de baixo posto (o cenário estático/fundo).
* **$S$ (Sparse)**: Representa a matriz esparsa, contendo as variações rápidas e não correlacionadas (os objetos em movimento/primeiro plano).

---

## Pré-requisitos e Instalação

Para rodar o projeto, é necessário ter o Python instalado em sua máquina. Além disso, o projeto faz uso das bibliotecas **NumPy, Matplotlib e OpenCV** (a biblioteca `time` é nativa do Python).

Caso não tenha as dependências instaladas, basta executar o seguinte comando no seu terminal para instalá-las automaticamente:

```bash
pip install -r requirements.txt

```

---

## Como Usar

### 1. Configurando o Vídeo de Entrada

Por padrão, o algoritmo buscará o vídeo a ser processado na pasta `data/`, sob o nome de arquivo `videoplayback`.

Caso queira utilizar o seu próprio vídeo, basta substituí-lo na pasta `data/` garantindo que o arquivo mantenha o mesmo nome.

> **Dica:** Dê preferência a vídeos curtos para evitar que o tempo de processamento da SVD seja excessivamente longo.

### 2. Executando o Código

Os códigos executáveis estão localizados na pasta `src/`. Você pode escolher qual abordagem deseja rodar:

* **`svd_manual.py`**: Implementação algorítmica da SVD construída "do zero", utilizando o **Método de Gauss-Jacobi**.
* **`svd.py`**: Implementação otimizada que faz uso da função nativa `numpy.linalg.svd` da biblioteca NumPy.

*Nota: Na pasta `src/` também encontra-se o arquivo `grayscale.py`. Trata-se de um módulo auxiliar importado pelos códigos principais para converter os frames do vídeo para escala de cinza, reduzindo a dimensionalidade dos dados e diminuindo o tempo de processamento.*

---

## Saídas e Resultados

Após a execução, todos os resultados gerados poderão ser encontrados na pasta `processed_data/`. Os arquivos de saída incluem:

**Vídeos gerados:**

* O vídeo original convertido para escala de cinza (*grayscale*).
* O vídeo evidenciando apenas os **objetos em movimento** (Matriz $S$).
* O vídeo evidenciando apenas o **cenário de fundo** (Matriz $L$), demonstrando a eficácia da separação.

**Métricas e Gráficos:**
Além dos vídeos, o código gera gráficos para a análise de desempenho e comportamento da SVD:

* **Decaimento dos Valores Singulares**
* **Variância Acumulada**
* **Erro de Reconstrução**
