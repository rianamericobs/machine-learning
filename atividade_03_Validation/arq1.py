import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC

# Carregamento e preparação dos dados
print("Carregando e preparando os dados...")
df = pd.read_csv('abalone_dataset.csv')

# Separando as características (X) da resposta (y)
y = df['type']
X = df.drop('type', axis=1)

# Transformando a coluna de texto 'Sex' em colunas numéricas (One-Hot Encoding)
X = pd.get_dummies(X, columns=['sex'], dtype=int)

# Padronizando as escalas para os algoritmos funcionarem bem (X_padronizado)
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)


# Validação Cruzada (Simulando o servidor):
#tentativa 1 com 62% de acurácia: modelo_knn = KNeighborsClassifier(n_neighbors=5)
#tentativa 2 com 64.43% de acurácia: modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
#tentativa 3 com 65.62% de acurácia: modelo_rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42)
#tentativa 4 com 65.42% de acurácia: modelo_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
#tentativa 5 com 66.99% de acurácia: modelo_svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nIniciando a validação cruzada... Isso pode levar alguns segundos.")
#resultados = cross_val_score(modelo_knn, X_padronizado, y, cv=kfold, scoring='accuracy')
#resultados = cross_val_score(modelo_rf, X_padronizado, y, cv=kfold, scoring='accuracy')
#resultados = cross_val_score(modelo_gb, X_padronizado, y, cv=kfold, scoring='accuracy')
#resultados = cross_val_score(modelo_svm, X_padronizado, y, cv=kfold, scoring='accuracy')



# Resultados
print(f"\nAcurácias de cada uma das 10 rodadas:\n {resultados}\n")
print(f"🏆 ACURÁCIA MÉDIA ESPERADA NO SERVIDOR: {resultados.mean() * 100:.2f}%")
