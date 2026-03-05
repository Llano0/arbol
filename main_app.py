import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuración de la página
st.set_page_config(page_title="Wine Classifier - Decision Trees", layout="wide")

st.title("🍷 Clasificador de Vinos con Árboles de Decisión")
st.markdown("""
Esta aplicación permite explorar cómo la **profundidad de un árbol** afecta la clasificación del famoso dataset 'Wine'.
""")

# 1. Carga de datos
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = load_data()

# --- Sidebar: Parámetros del Modelo ---
st.sidebar.header("Configuración del Modelo")

max_depth = st.sidebar.slider("Profundidad máxima del árbol (max_depth)", 1, 10, 3)
cv_folds = st.sidebar.number_input("Número de Folds para Validación Cruzada", 2, 10, 5)
test_size = st.sidebar.slider("Tamaño del set de prueba (%)", 10, 50, 20) / 100

# --- Procesamiento ---
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Inicializar y entrenar el modelo
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# --- Visualización de Resultados ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Métricas de Evaluación")
    
    # Validación Cruzada
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
    
    st.write(f"**Accuracy en Test:** `{accuracy_score(y_test, y_pred):.2f}`")
    st.write(f"**Promedio Validación Cruzada ({cv_folds} folds):** `{cv_scores.mean():.2f}`")
    
    # Reporte de Clasificación en formato tabla
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

with col2:
    st.subheader("🧩 Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax_cm, cmap="YlGnBu")
    st.pyplot(fig_cm)

# --- Visualización del Árbol ---
st.divider()
st.subheader("🌳 Estructura del Árbol de Decisión")
fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
plot_tree(clf, 
          feature_names=df.columns[:-1], 
          class_names=target_names, 
          filled=True, 
          rounded=True, 
          fontsize=12, 
          ax=ax_tree)
st.pyplot(fig_tree)

# Mostrar datos crudos si el usuario quiere
if st.checkbox("Mostrar Dataset Original"):
    st.dataframe(df)
