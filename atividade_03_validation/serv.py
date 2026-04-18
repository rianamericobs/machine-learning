"""
Atividade 03 - Envio de predicoes ao servidor.
Carrega as predicoes geradas pelo main.py e envia para o servidor de avaliacao.
"""

import pandas as pd
import requests

# Verifica se as predicoes ja foram geradas
try:
    predictions = pd.read_csv("predictions.csv")
    y_pred = predictions["prediction"].values
    print(f"\n - Predicoes carregadas de predictions.csv ({len(y_pred)} amostras)")
except FileNotFoundError:
    print("\n[ERRO] Arquivo predictions.csv nao encontrado.")
    print("Execute 'python main.py' primeiro para gerar as predicoes.")
    exit(1)

# Enviando previsoes realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

DEV_KEY = "algoWRIthm"

# json para ser enviado para o servidor
data = {
    "dev_key": DEV_KEY,
    "predictions": pd.Series(y_pred).to_json(orient="values"),
}

print(" - Enviando predicoes para o servidor...")
r = requests.post(url=URL, data=data)

print(" - Resposta do servidor:\n", r.text, "\n")
