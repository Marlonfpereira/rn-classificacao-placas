# Realização de testes de predição com o dataset e menu de verificação das imagens preditas
# Igor Gris, Marlon Pereira e Ronaldo Drecksler

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
import h5py

def get_description(value):
    descriptions = [
        'Speed limit (20km/h)',
        'Speed limit (30km/h)',
        'Speed limit (50km/h)',
        'Speed limit (60km/h)',
        'Speed limit (70km/h)',
        'Speed limit (80km/h)',
        'End of speed limit (80km/h)',
        'Speed limit (100km/h)',
        'Speed limit (120km/h)',
        'No passing',
        'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection',
        'Priority road',
        'Yield',
        'Stop',
        'No vehicles',
        'Vehicles over 3.5 metric tons prohibited',
        'No entry',
        'General caution',
        'Dangerous curve to the left',
        'Dangerous curve to the right',
        'Double curve',
        'Bumpy road',
        'Slippery road',
        'Road narrows on the right',
        'Road work',
        'Traffic signals',
        'Pedestrians',
        'Children crossing',
        'Bicycles crossing',
        'Beware of ice/snow',
        'Wild animals crossing',
        'End of all speed and passing limits',
        'Turn right ahead',
        'Turn left ahead',
        'Ahead only',
        'Go straight or right',
        'Go straight or left',
        'Keep right',
        'Keep left',
        'Roundabout mandatory',
        'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons',
    ]
    return descriptions[value]

try:
    with open('predictions.txt', 'r') as f:
        pass
except FileNotFoundError:
    # Carregar o modelo treinado
    # Se o modelo já estiver carregado em uma variável, você pode ignorar esta linha
    model = load_model('modelo.h5')

    # Supondo que X_test e y_test são seus dados de teste e rótulos
    # X_test: features do conjunto de teste
    # y_test: rótulos verdadeiros do conjunto de teste


    with h5py.File('dataset_ts_light_version.hdf5', 'r') as hf:
        x_test = np.array(hf['x_test'])
        y_test = to_categorical(np.array(hf['y_test']))


    # Obter as previsões do modelo
    y_pred = model.predict(x_test)

    # Verificar se os rótulos verdadeiros são one-hot encoded ou classes
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        # Caso os rótulos verdadeiros estejam em formato one-hot encoded, converta-os para classes
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_test_classes = y_test

    # Converter as previsões para classes
    y_pred_classes = np.argmax(y_pred, axis=1)


    for i in range(243):
        with open('predictions.txt', 'a') as f:
            f.write(f"{i}")
            f.write("-------------\n")
            f.write("Prediction: {}\n".format(y_pred_classes[i]))
            f.write("True label: {}\n".format(y_test_classes[i]))
            f.write("Description of prediction: {}\n".format(get_description(y_pred_classes[i])))
            f.write("Description of true label: {}\n".format(get_description(y_test_classes[i])))
        

while True:
    print("Insira o número do teste que deseja checar: (-1 para sair)")
    try:
        num = int(input())
        if num == -1: 
            break
        if 0 <= num <= 242:
            with h5py.File('dataset_ts_light_version.hdf5', 'r') as hf:
                X_test = np.array(hf['x_test'])
                plt.imshow(X_test[num].astype('uint8'))
                plt.axis('off') 
                plt.show()
        else:
            print("Invalid number")
    except ValueError:
        print("Invalid input")
