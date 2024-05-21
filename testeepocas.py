# Teste de quantidade de épocas adequadas aos parâmetros obtidos em grid search
# Igor Gris, Marlon Pereira e Ronaldo Drecksler

import h5py
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm 

# Carregar dados
with h5py.File('dataset_ts_original.hdf5', 'r') as hf:
    X_train = np.array(hf['x_train']) / 255.0
    Y_train = to_categorical(np.array(hf['y_train']))
    X_test = np.array(hf['x_test']) / 255.0
    Y_test = to_categorical(np.array(hf['y_test']))

# Separar uma parte dos dados de treinamento para validação
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Definir a função de criação do modelo
def create_model(optimizer='adam', learn_rate=0.001, init_mode='uniform', 
                 conv_layers=1, conv_filters=32, dense_layers=1, dense_units=128):
    model = Sequential()
    
    # Camadas convolucionais

    model.add(Conv2D(conv_filters, (3, 3), activation='relu', kernel_initializer=init_mode, input_shape=(48, 48, 3)))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(2,conv_layers):
        model.add(Conv2D(conv_filters, (3, 3), activation='relu', kernel_initializer=init_mode))
        model.add(MaxPooling2D((2, 2)))
        


    model.add(Flatten())
    
    # Camadas densas
    for _ in range(dense_layers):
        model.add(Dense(dense_units, activation='relu', kernel_initializer=init_mode))
    
    # Camada de saída
    model.add(Dense(43, activation='softmax', kernel_initializer=init_mode))
    
    optimizer_instance = None
    if optimizer == 'SGD':
        optimizer_instance = SGD(learning_rate=learn_rate)
    elif optimizer == 'Adam':
        optimizer_instance = Adam(learning_rate=learn_rate)
    elif optimizer == 'RMSprop':
        optimizer_instance = RMSprop(learning_rate=learn_rate)
    
    model.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Envolver o modelo com KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Definir os parâmetros do grid search
# Best: 0.9446090459823608 using {'batch_size': 50, 'conv_filters': 32, 'conv_layers': 1, 'dense_layers': 2, 'dense_units': 256, 'epochs': 5, 'learn_rate': 0.001, 'optimizer': 'Adam'}
param_grid = { # 768
    'batch_size': [50],
    'epochs': [10, 20, 30, 40, 50],
    'optimizer': ['Adam'],
    'learn_rate': [0.001],
    'conv_layers': [1],
    'conv_filters': [32],
    'dense_layers': [2],
    'dense_units': [256]
}

# Executar o grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=1)
grid_result = grid.fit(X_train, Y_train, validation_data=(X_val, Y_val))


# Imprimir os resultados
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print()
    print(f"{mean} ({stdev}) with: {param}")

