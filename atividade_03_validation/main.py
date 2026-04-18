"""
Atividade 03 - Avaliacao de Classificadores
Pipeline completo: EDA, pre-processamento, treinamento, validacao e selecao do melhor modelo.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATASET_PATH = "abalone_dataset.csv"
APP_PATH = "abalone_app.csv"
RANDOM_STATE = 42

# =============================================================================
# 1. ANALISE EXPLORATORIA DOS DADOS (EDA)
# =============================================================================
print("=" * 60)
print("ETAPA 1: ANALISE EXPLORATORIA DOS DADOS")
print("=" * 60)

data = pd.read_csv(DATASET_PATH)

print(f"\nDimensoes do dataset: {data.shape}")
print(f"\nPrimeiras linhas:\n{data.head()}")
print(f"\nTipos de dados:\n{data.dtypes}")
print(f"\nValores ausentes:\n{data.isnull().sum()}")
print(f"\nEstatisticas descritivas:\n{data.describe()}")
print(f"\nDistribuicao das classes (type):\n{data['type'].value_counts().sort_index()}")
print(f"\nProporcao das classes:\n{data['type'].value_counts(normalize=True).sort_index()}")
print(f"\nDistribuicao de sex:\n{data['sex'].value_counts()}")

numeric_cols = data.select_dtypes(include=[np.number]).columns.drop("type")
print(f"\nCorrelacao entre features numericas:\n{data[numeric_cols].corr().round(2)}")

# =============================================================================
# 2. PRE-PROCESSAMENTO
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 2: PRE-PROCESSAMENTO")
print("=" * 60)

feature_cols = [
    "sex", "length", "diameter", "height",
    "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",
]

X = data[feature_cols]
y = data["type"]

categorical_features = ["sex"]
numerical_features = [c for c in feature_cols if c != "sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)

print(f"Features categoricas: {categorical_features}")
print(f"Features numericas: {numerical_features}")
print("Preprocessamento: StandardScaler + OneHotEncoder(drop='first')")

# =============================================================================
# 3. TREINAMENTO E COMPARACAO DE MULTIPLOS CLASSIFICADORES
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 3: COMPARACAO DE CLASSIFICADORES (Stratified 10-Fold CV)")
print("=" * 60)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = {}
print(f"\n{'Modelo':<25} {'Acuracia Media':>15} {'Desvio Padrao':>15}")
print("-" * 57)

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    results[name] = scores
    print(f"{name:<25} {scores.mean():>15.4f} {scores.std():>15.4f}")

# =============================================================================
# 4. TUNING DE HIPERPARAMETROS
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 4: TUNING DE HIPERPARAMETROS (GridSearchCV)")
print("=" * 60)

# Selecionar os 3 melhores modelos
ranking = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
top_models = [name for name, _ in ranking[:3]]
print(f"\nTop 3 modelos para tuning: {top_models}")

param_grids = {
    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10, 50],
        "classifier__kernel": ["rbf", "poly"],
        "classifier__gamma": ["scale", "auto"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
    },
    "Decision Tree": {
        "classifier__max_depth": [None, 5, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__criterion": ["gini", "entropy"],
    },
}

tuned_results = {}

for name in top_models:
    print(f"\nTuning: {name}...")
    model = models[name]
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    grid = GridSearchCV(
        pipe,
        param_grids[name],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    tuned_results[name] = grid
    print(f"  Melhor acuracia: {grid.best_score_:.4f}")
    print(f"  Melhores parametros: {grid.best_params_}")

# =============================================================================
# 5. SELECAO DO MELHOR MODELO
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 5: SELECAO DO MELHOR MODELO")
print("=" * 60)

best_name = max(tuned_results, key=lambda k: tuned_results[k].best_score_)
best_grid = tuned_results[best_name]

print(f"\nMelhor modelo: {best_name}")
print(f"Acuracia (CV): {best_grid.best_score_:.4f}")
print(f"Parametros: {best_grid.best_params_}")

# O GridSearchCV com refit=True ja treinou no dataset completo
best_pipeline = best_grid.best_estimator_

# =============================================================================
# 6. GERACAO DE PREDICOES
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 6: GERACAO DE PREDICOES PARA abalone_app.csv")
print("=" * 60)

data_app = pd.read_csv(APP_PATH)
X_app = data_app[feature_cols]
y_pred = best_pipeline.predict(X_app)

print(f"\nPredicoes geradas: {len(y_pred)} amostras")
print(f"Distribuicao das predicoes:\n{pd.Series(y_pred).value_counts().sort_index()}")

# Salvar predicoes em CSV para referencia
predictions_df = pd.DataFrame({"prediction": y_pred})
predictions_df.to_csv("predictions.csv", index=False)
print("\nPredicoes salvas em predictions.csv")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"\n{'Modelo':<25} {'Acuracia CV':>15}")
print("-" * 42)
for name, scores in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
    marker = " <-- baseline" if name == "Decision Tree" else ""
    print(f"{name:<25} {scores.mean():>15.4f}{marker}")
print(f"\n>>> Modelo selecionado: {best_name} (acuracia CV: {best_grid.best_score_:.4f})")
print("\nPara enviar as predicoes ao servidor, execute: python serv.py")
