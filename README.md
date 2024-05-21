# Classificação de Placas de Trânsito com Redes Neurais

[Dataset Utilizado](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-1-million-images-for-classification) | São usados os arquivos `dataset_ts_light_version.hdf5` e `dataset_ts_original.hdf5`

Todos os códigos podem ser executados com a linha de comando `python3 <nome do arquivo>`, sendo eles os seguintes:

- `BestFitParams.py` > Execução da GridSearch para a busca da melhor combinação de parâmetros.
- `BestFitEpochs.py` > Execução da GridSearch para a busca da melhor quantidade de épocas, dados os parâmetros encontrados com `BestFitParams.py`.
- `generateModel.py` > São usados os parâmetros e quantidades de épocas encontradas para gerar um modelo.
- `generateConfusionMatrix.py` > Geração da matriz de confusão com o modelo criado.
- `imgTest.py` > Escrita de predições com rótulos em um arquivo, juntamente com a opção de verificar a imagem da placa sendo predita.