import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout


def build_and_train_model(X_train, y_train, X_test):
    """
    Build and train a deep neural network with regularization.

    Parameters:
        X_train (np.ndarray): Training inputs of shape (n_samples, 1)
        y_train (np.ndarray): Training targets of shape (n_samples,)
        X_test (np.ndarray): Test inputs of shape (n_samples, 1)

    Returns:
        np.ndarray: Predictions for X_test
    """
    
    # TODO: Build a deep model with regularization (L1, L2, or Dropout)
    
    modelo = Sequential()
    # fique em duvida de qual usar, dai usei os dois, porem apresentou erro de aprendizado
    dropout = 0.1
    l2_reg = 0.0


    # Camadas ocultas da rede neural
    modelo.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_reg)))

    modelo.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    modelo.add(Dropout(dropout))

    modelo.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    modelo.add(Dropout(dropout))

    modelo.add(Dense(16, activation='relu', kernel_regularizer=l2(l2_reg)))
    modelo.add(Dropout(dropout))
    
    modelo.add(Dense(8, activation='relu', kernel_regularizer=l2(l2_reg)))
    modelo.add(Dropout(dropout))
    
    modelo.add(Dense(4, activation='relu', kernel_regularizer=l2(l2_reg)))
    modelo.add(Dropout(dropout))
        
    
    # camada de saida (nao oculta) para regressao
    modelo.add(Dense(1, activation='linear'))

    modelo.compile(optimizer=Adam(), loss='mean_squared_error')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    modelo.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    modelo.add(Dense(1, activation='linear'))

    modelo.compile(optimizer=Adam(), loss='mean_squared_error')

    y_predito = modelo.predict(X_test)

    return y_predito
