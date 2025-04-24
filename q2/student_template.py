import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_and_train_model(X_train, y_train, X_test):
    """
    Builds and trains a DNN model for multivariable regression.

    Parameters:
        X_train (pd.DataFrame): Training input data with mixed types.
        y_train (np.ndarray): Regression targets.
        X_test (pd.DataFrame): Test input data.

    Returns:
        np.ndarray: Predicted values for X_test.
    """
    # organizando os dados para ser mais facil manipulalos para treinar o modelo
    n_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorias_features = X_train.select_dtypes(include=['object']).columns

    preprocessamento = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), n_features),
            ('cat', OneHotEncoder(), categorias_features)
        ])
    
    # Transforming the data
    X_treino_processado = preprocessamento.fit_transform(X_train)
    X_teste_processado = preprocessamento.transform(X_test)

    # Defining a sequential neural network model
    modelo = Sequential()

    # Hidden layers of the neural network
    modelo.add(Dense(128, input_dim=X_treino_processado.shape[1], activation='relu'))  # Hidden layer 1
    modelo.add(Dense(64, activation='relu'))  # Hidden layer 2
    modelo.add(Dense(32, activation='relu'))  # Hidden layer 3
    modelo.add(Dense(16, activation='relu'))  # Hidden layer 4

    # Output layer (non-hidden) for regression
    modelo.add(Dense(1, activation='linear'))  

    # Compile the model
    modelo.compile(optimizer=Adam(), loss='mean_squared_error')

    # Train the model
    modelo.fit(X_treino_processado, y_train, epochs=50, batch_size=32, verbose=1)

    # Get predictions
    y_predito = modelo.predict(X_teste_processado)

    return y_predito