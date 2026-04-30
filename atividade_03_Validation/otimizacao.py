import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

print("Preparando os dados com ENGENHARIA DE ATRIBUTOS...")
df = pd.read_csv('abalone_dataset.csv')

y = df['type']
X = df.drop('type', axis=1)

# Criando novas colunas matemáticas (Nomes Corrigidos)
# 1. Volume aproximado do molusco
X['volume'] = X['length'] * X['diameter'] * X['height']

# 2. Peso da água/sangue
X['water_loss'] = X['whole_weight'] - (X['shucked_weight'] + X['viscera_weight'] + X['shell_weight'])

# 3. Proporção da concha
X['shell_ratio'] = X['shell_weight'] / (X['whole_weight'] + 0.0001)

# 4. Densidade aproximada
X['density'] = X['whole_weight'] / (X['volume'] + 0.0001)

# Tratando a coluna de texto 'sex'
X = pd.get_dummies(X, columns=['sex'], dtype=int)

# Padronizando todas as colunas (as originais e as novas)
scaler = StandardScaler()
X_transformado = scaler.fit_transform(X)

# Usando o SVM com ajuste fino
modelo_svm = SVC(C=15, gamma='scale', kernel='rbf', random_state=42)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("Avaliando o modelo com as novas características...")
resultados = cross_val_score(modelo_svm, X_transformado, y, cv=kfold, scoring='accuracy')

print(f"\nAcurácias das 10 rodadas:\n {resultados}")
print(f"🏆 ACURÁCIA MÉDIA COM ENGENHARIA: {resultados.mean() * 100:.2f}%")