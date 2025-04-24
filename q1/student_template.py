import numpy as np

def train_perceptron(X, y, epochs=50, lr=0.1):
    """
    Train a Perceptron model on data X with labels y.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Labels (-1 or 1) of shape (n_samples,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        weights (np.ndarray): Trained weights including bias term.
    """
    # Add your implementation below
    # pass
    
    # print(f'{X}\n{y}')
    pesos = np.zeros(X.shape[1])
    # inicializa um vetor de pesos com todas as possicoes recebendo zeros

    for epoca in range(epochs):
        """
        primeiro forIn itera sobre as epocas para treinar o modelo, 
        logo cada dado em X é treinado epoca vezes
        """
        for i in range(len(X)):
            # o segundo forIn itera sobre os dados de entrada e os pesos
            predicao = np.sign(np.dot(X[i], pesos))
            if predicao!=y[i]:
                """
                se a predicao for diferente do valor esperado, atualiza
                os pesos de acordo com a regra de atualizacao do perceptron
                para que na proxima epoca o modelo TENTE prever corretamente
                """
                pesos+=lr*y[i]*X[i]
    return pesos