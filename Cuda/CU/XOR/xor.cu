#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
__global__ sempre retorna void
*/
__global__ void loss_fn() {
    
}

__global__ void forward_fn() {
/*
Z1 = X * W1 + b1
A1 = activation(Z1)

Z2 = A1 * W2 + b2
A2 = activation(Z2)

Z3 = A2 * W3 + b3
y_hat = sigmoid(Z3)
*/
}

__global__ void backpropagation_fn() {

}


// gera valores aleatórios entre -0.5 e 0.5
float rand_weight() {
    return ((float)rand() / RAND_MAX) - 0.5f;
}

int main() {
    // SETTINGS
    // aleatoriedade
    srand(time(NULL));
    // DATASET ---------------------------------------------
    // criando o nosso conjunto de dados X e y
    // 4 linhas e 2 colunas
    float X_cpu[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    // bytes que precisamos para X
    size_t size_X = 4 * 2 * sizeof(float);
    // 4 linhas e 1 coluna
    float y_cpu[4][1] = {
        {0},
        {1},
        {1},
        {0}
    };
    // bytes que precisamos para y
    size_t size_y = 4 * 1 * sizeof(float);
    // cria o vetor de predict da rede
    float y_hat_cpu[4][1];

    // criando os ponteiros para enviar para a GPU
    float *X, *y, *y_hat;

    // pedindo para liberar o espaço na memoria
    cudaMalloc(&X, size_X);  // X é uma matrix de 4x2
    cudaMalloc(&y, size_y);  // y é uma matrix de 4x1
    cudaMalloc(&y_hat, size_y);

    // enviando para a GPU
    cudaMemcpy(X, X_cpu, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_cpu, size_y, cudaMemcpyHostToDevice);
    cudaMemcpy(y_hat, y_hat_cpu, size_y, cudaMemcpyHostToDevice);

    // INICIALIZA OS PESOS DA REDE ------------------------------
    // a nossa rede tera 3 camadas com 4 neuronios na primeira, 2 na segunda e 1  na ultima
    // irei fazer de forma linear

    // camada 1
    float W1_cpu[2][4];
    float b1_cpu[4];

    // camada 2
    float W2_cpu[4][2];
    float b2_cpu[2];

    // camada 3
    float W3_cpu[2][1];
    float b3_cpu[1];

    // dando valores aos pesos
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 4; j++)
            W1_cpu[i][j] = rand_weight();

    for(int i = 0; i < 4; i++)
        b1_cpu[i] = rand_weight();


    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 2; j++)
            W2_cpu[i][j] = rand_weight();

    for(int i = 0; i < 2; i++)
        b2_cpu[i] = rand_weight();


    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 1; j++)
            W3_cpu[i][j] = rand_weight();

    b3_cpu[0] = rand_weight();

    // alocando na gpu
    float *W1, *b1;
    float *W2, *b2;
    float *W3, *b3;

    cudaMalloc(&W1, 2 * 4 * sizeof(float));
    cudaMalloc(&b1, 4 * sizeof(float));

    cudaMalloc(&W2, 4 * 2 * sizeof(float));
    cudaMalloc(&b2, 2 * sizeof(float));

    cudaMalloc(&W3, 2 * 1 * sizeof(float));
    cudaMalloc(&b3, 1 * sizeof(float));

    //movendo para a gpu
    cudaMemcpy(W1, W1_cpu, 2*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, b1_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(W2, W2_cpu, 4*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, b2_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(W3, W3_cpu, 2*1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b3, b3_cpu, 1*sizeof(float), cudaMemcpyHostToDevice);
    // MONTA A REDE ---------------------------------------------

    // TREINA A REDE --------------------------------------------

    // VÊ A LOSS ------------------------------------------------

    // PREDITA --------------------------------------------------









    // LIBERANDO MEMORIA
    // DATASET
    // liberar memória
    // cudaFree(X);
    // cudaFree(y);
    // cudaFree(y_hat);

    // ---
    return 0;
}