## Objetivo

Construir e validar modelos preditivos para classificar exemplares de Abalone em tres classes (1, 2 e 3), garantindo o poder de generalizacao do modelo atraves de tecnicas de validacao.

## Etapas

### 1. Analise Exploratoria dos Dados (EDA)
- Carregar `abalone_dataset.csv` e inspecionar as primeiras linhas
- Verificar valores ausentes e tipos de dados
- Analisar o balanceamento das classes (coluna `type`)
- Estatisticas descritivas das features numericas
- Analisar correlacao entre as variaveis

### 2. Pre-processamento
- Codificar a variavel categorica `sex` (One-Hot Encoding)
- Normalizar/padronizar as features numericas (StandardScaler)
- Engenharia de features: criar razoes entre colunas de peso, se util

### 3. Treinamento e Comparacao de Multiplos Classificadores
- Treinar os seguintes modelos usando **Stratified K-Fold Cross-Validation (k=10)**:
  - Decision Tree (baseline)
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
- Comparar a acuracia media de cada modelo via cross-validation

### 4. Tuning de Hiperparametros
- Aplicar `GridSearchCV` nos 2-3 melhores modelos
- Avaliar com cross-validation para evitar overfitting

### 5. Selecao do Melhor Modelo e Validacao Final
- Selecionar o modelo com melhor acuracia validada
- Treinar o modelo final no dataset completo (`abalone_dataset.csv`)

### 6. Geracao de Predicoes e Envio ao Servidor
- Aplicar o modelo treinado em `abalone_app.csv`
- Enviar as predicoes para `https://aydanomachado.com/mlclass/03_Validation.php`

