# Geração do modelo com os hiperparâmetros encontrados por GridSearch
# Igor Gris, Marlon Pereira e Ronaldo Drecksler

import h5py
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

ot = Adam(learning_rate=0.001)


with h5py.File('dataset_ts_original.hdf5', 'r') as hf:
    X_val = np.array(hf['x_validation'])/255.0
    Y_val = to_categorical(np.array(hf['y_validation']))
    X_train = np.array(hf['x_train'])/255.0
    Y_train = to_categorical( np.array(hf['y_train']))
    X_test = np.array(hf['x_test'])/255.0
    Y_test = to_categorical(np.array(hf['y_test']))


## 1 => (1,0,0)

# Definindo a arquitetura do modelo
model = Sequential()

# Primeira camada convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))

# Camada de achatamento
model.add(Flatten())

# Camada totalmente conectada
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

# Camada de saída
model.add(Dense(43, activation='softmax'))

# Compilando o modelo
model.compile(optimizer=ot, loss='categorical_crossentropy', metrics=['accuracy'])

# Exibindo a arquitetura do modelo
history = model.fit(X_train, Y_train, epochs=1, batch_size=50, validation_data=(X_val, Y_val))

model.save('modelo.h5')

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, Y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
