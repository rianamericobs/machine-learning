import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre abalones')
data = pd.read_csv('abalone_dataset.csv')

print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo abalone_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['sex', 'length', 'diameter', 'height', 
                'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
X = data[feature_cols]
y = data.Type

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('abalone_app.csv')
data_app = data_app[feature_cols]
y_pred = modelo.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "algoWRIthm"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")