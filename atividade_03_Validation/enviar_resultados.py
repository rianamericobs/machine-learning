import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Preparando os dados de treino
print("Treinando o modelo com Engenharia de Atributos...")
df_treino = pd.read_csv('abalone_dataset.csv')

y_treino = df_treino['type']
X_treino = df_treino.drop('type', axis=1)

# Aplicando a Engenharia de Atributos no treino
X_treino['volume'] = X_treino['length'] * X_treino['diameter'] * X_treino['height']
X_treino['water_loss'] = X_treino['whole_weight'] - (X_treino['shucked_weight'] + X_treino['viscera_weight'] + X_treino['shell_weight'])
X_treino['shell_ratio'] = X_treino['shell_weight'] / (X_treino['whole_weight'] + 0.0001)
X_treino['density'] = X_treino['whole_weight'] / (X_treino['volume'] + 0.0001)

# Convertendo a coluna de sexo e padronizando
X_treino = pd.get_dummies(X_treino, columns=['sex'], dtype=int)
scaler = StandardScaler()
X_treino_padronizado = scaler.fit_transform(X_treino)

# Treinando o modelo
modelo_final = SVC(C=15, gamma='scale', kernel='rbf', random_state=42)
modelo_final.fit(X_treino_padronizado, y_treino)

# Preparando os dados de teste
print("Lendo o arquivo de teste e aplicando as mesmas transformações...")
df_teste = pd.read_csv('abalone_app.csv')

# Aplicando a Engenharia de Atributos no teste
df_teste['volume'] = df_teste['length'] * df_teste['diameter'] * df_teste['height']
df_teste['water_loss'] = df_teste['whole_weight'] - (df_teste['shucked_weight'] + df_teste['viscera_weight'] + df_teste['shell_weight'])
df_teste['shell_ratio'] = df_teste['shell_weight'] / (df_teste['whole_weight'] + 0.0001)
df_teste['density'] = df_teste['whole_weight'] / (df_teste['volume'] + 0.0001)

X_teste = pd.get_dummies(df_teste, columns=['sex'], dtype=int)

# Garantindo as colunas idênticas
X_teste = X_teste.reindex(columns=X_treino.columns, fill_value=0)

X_teste_padronizado = scaler.transform(X_teste)

print("Fazendo as previsões oficiais...")
y_pred = modelo_final.predict(X_teste_padronizado)

# Enviando para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

DEV_KEY = "algoWRIthm"

data = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}

print("Enviando as previsões para o servidor...")

try:
    r = requests.post(url=URL, data=data)
    print("\n--- Resposta do servidor ---\n", r.text, "\n")
except Exception as e:
    print(f"\nErro ao tentar conectar com o servidor: {e}")