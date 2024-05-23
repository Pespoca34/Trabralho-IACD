import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

file_path = 'C:\Development\Trabalho IACD - 2\hcc_dataset.csv'
hcc_data = pd.read_csv(file_path)

#! Selecione uma coluna numérica específica para a transformação log1p, substituindo 'Numeric_Column' pelo nome correto da coluna
if 'Numeric_Column' in hcc_data.columns:
    hcc_data['New_Feature'] = np.log1p(hcc_data['Numeric_Column'])  #? Aplica log1p na coluna correta


#* Mostra um pouco do dataset
print("First few rows of the dataset:")
print(hcc_data.head())

print("\nSummary statistics:")
print(hcc_data.describe(include='all'))

print("\nMissing values in the dataset:")
print(hcc_data.isnull().sum())

target_column = hcc_data.columns[-1]
print(f"\nDistribution of the target variable '{target_column}':")
print(hcc_data[target_column].value_counts())

#! Separa features e target, com o intuito de não generalizar apenas
#? Features são os valores de input, usados para fazer o predict
#? Target é o output que você deseja fazer o predict
X = hcc_data.drop(columns=[target_column])
y = hcc_data[target_column]

#! Diferencia as colunas entre numericas e categoricas
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

#? Prepocessing de valores numericos de data, ignora os nulos 
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#? Prepocessing das classes do arquivo csv
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#? Combina os passos de prepocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#! RandomForest com balanceamento de classe
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

#! KNeighborsClassifier com possibilidade de ajuste de hiperparâmetros
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Ajuste o número de vizinhos se necessário
])

#? Divide o modelo em teste e traino = 70% -> Treino, 30 -> Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#! Identifica a importancia das características, para identificar se é melhor eliminar alguma de menor impacto
from sklearn.utils import permutation_importance

def permutation_importance(model, X_test, y_test, scoring='accuracy'):
  # inicializa um array vazio para guardar as pontuações de importanciaI
  importances = np.zeros(X_test.shape[1])

  # Itera entre features
  for i in range(X_test.shape[1]):
    # mistura uma copia dos features data
    X_test_shuffled = X_test.copy()
    X_test_shuffled[:, i] = np.random.permutation(X_test_shuffled[:, i])

    # Predice utilizando os dados misturados do feature data
    y_pred_shuffled = model.predict(X_test_shuffled)

    # Calcula a diferença de desempenho (original vs misturado)
    importance = scoring(y_test, y_pred_shuffled) - scoring(y_test, model.predict(X_test))
    importances[i] = importance

  # Return the importance scores
  return importances


#! Treino e avaliação para RandomForest
rf_pipeline.fit(X_train, y_train)
y_preds_rf = rf_pipeline.predict(X_test)

print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_preds_rf))
print("ROC AUC:", roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds_rf))
print("Classification Report:\n", classification_report(y_test, y_preds_rf))

#! Identifica a importancia das características, para identificar se é melhor eliminar alguma de menor valor (2º metodo)
rf_pipeline.fit(X_train, y_train)

# Extrai pontuações de importância dos fetures do modelo treinado
feature_importances = rf_pipeline.steps[-1][1].feature_importances_

# Print feature names and importances
print("Feature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

#! Treino e avaliação para KNN
knn_pipeline.fit(X_train, y_train)
y_preds_knn = knn_pipeline.predict(X_test)

print("\nKNN Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_preds_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds_knn))
print("Classification Report:\n", classification_report(y_test, y_preds_knn))

#? Histograma das colunas numéricas
hcc_data[numerical_cols].hist(figsize=(14, 14), bins=20)
plt.show()

#* Informação extra, mas não necessária
# print("\nInterpretation of Results:")
# print("The Random Forest Classifier has been trained and evaluated. The accuracy and ROC AUC scores provide a measure of the model's performance.")
# print("The confusion matrix and classification report give detailed insights into the performance metrics such as precision, recall, and F1-score.")
# print("Further analysis and hyperparameter tuning could be conducted to improve the model performance. Additionally, other classifiers and techniques can be explored for better results.")

#? Precision = True positive / True postivie + False positive
#? Recall = True positive / True positive + False negative
#? F1-score = 2 * ((Precision * recall) / (precison + recall))
#? Support show the viewed data
