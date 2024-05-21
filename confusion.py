# Criação da matriz de confusão com o modelo treinado
# Igor Gris, Marlon Pereira e Ronaldo Drecksler

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from keras.utils import to_categorical
import h5py

# Carregar o modelo treinado
# Se o modelo já estiver carregado em uma variável, você pode ignorar esta linha
model = load_model('modelo.h5')

# Supondo que X_test e y_test são seus dados de teste e rótulos
# X_test: features do conjunto de teste
# y_test: rótulos verdadeiros do conjunto de teste


with h5py.File('dataset_ts_light_version.hdf5', 'r') as hf:
    x_test = np.array(hf['x_test'])/255.0
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


# Calcular a matriz de confusão
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Visualizar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
