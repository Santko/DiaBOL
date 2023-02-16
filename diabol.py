# Esta aplicación ha sido desarrollada para intentar detectar si una persona tiene diabetes utilizando inteligencia artificial y python.
# Esta aplicación no realiza diagnosticos Los resultados evaluados por Inteligencia Artificial basados ​​en una Muestra de entreanamiento son solo recomendaciones!


# Estas son las librerias que se han utilizado para desarrollar la aplicación
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Creación de Titulos y Subtitulos
st.write("""
# DiaBOL Detección de diabetes
Con Inteligencia Artificial, intenta detectar si una paciente tiene diabetes o no.
¡NO ES UN DIAGNÓSTICO! Los resultados evaluados por Inteligencia Artificial basados ​​en una muestra de entrenamiento son solo recomendaciones.
""")
# Creación de Titulos y Subtitulos
img = Image.open(r'C:\Users\mario\.conda\envs\primerpy\DiaBOL\test image.png')
st.image(img, caption='AI', use_column_width=True)

# Esta es la muestra de entrenamiento Recibe datos para el modelo de machine learning (con RandomForestClassifier) para su entrenamiento
df = pd.read_csv(r'C:\Users\mario\.conda\envs\primerpy\DiaBOL\DiabetesDataset.csv')
# Creación de Subtitulos
st.subheader('Conjunto de datos para entrenamiento')
# Proyecta los datos en el conjunto de datos como una tabla en la aplicación web
st.dataframe(df)
# Muestra alguna información estadística (máximo, mínimo, etc.) de los datos
st.write(df.describe())
# Muestra datos como gráficos
chart = st.bar_chart(df)

# Divide los datos en variables independientes 'X' y dependientes 'Y'
X = df.iloc[:, 0:8].values  # La variable x contiene todas las características del conjunto de datos (es decir, todas las columnas excepto Resultado).
Y = df.iloc[:, -1].values  # La variable Y contiene la columna Resultado.
# Divide el 75% de los datos en el conjunto de datos para entrenar el modelo y el 25% para las pruebas
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Parrafo para conocer los valores del paciente, para saber si tiene diabetes o no.
def get_user_input():
    embarazos = st.sidebar.slider('Cantidad de Embarazos', 0, 15, 1)  # el número de veces que una paciente estuvo embarazada (mínimo = 0, máximo = 15, predeterminado = 1)
    glucosa = st.sidebar.slider('Glucosa en sangre', 0, 199, 110) # valor de glucosa en sangre (mínimo = 0, máximo = 199, predeterminado = 110)
    presion_arterial = st.sidebar.slider('Presion Arterial', 0, 140, 72) # valor de presión arterial (min = 0, máximo = 140, predeterminado = 72)
    grosor_piel = st.sidebar.slider('Grosor en la piel', 0, 99, 23) # valor del grosor de la piel (mínimo = 0, máximo = 99, predeterminado = 23)
    insulina = st.sidebar.slider('Nivel de Insulina', 0, 126, 100) # valor de insulina (min = 0, máximo = 126, predeterminado = 100)
    imc = st.sidebar.slider('Indice de Masa Corporal', 0.0, 50.0, 21.5) # Valor de IMC Indice de Masa Corporal (mínimo = 0,0, máximo = 50,0, predeterminado = 21,5)
    afd = st.sidebar.slider('Antecedentes Familiares Diabetes', 0.0, 2.49, 0.3725) # Antecedentes familiares de diabetes (mínimo = 0,0, máximo = 2,49, predeterminado = 0,3725)
    edad = st.sidebar.slider('Edad', 18, 99, 30) # valor de edad (mínimo = 18, máximo = 99, predeterminado = 30)

    # Registra los valores recibidos del usuario en una estructura de diccionario (diccionario) como pares clave-valor
    user_data = {'embarazos': embarazos, 'glucosa': glucosa, 'presion_arterial': presion_arterial, 'grosor_piel': grosor_piel, 'insulina': insulina, 'imc': imc, 'afd': afd, 'edad': edad}
    # Conversión de datos de usuario a marco de datos
    features = pd.DataFrame(user_data, index=[0])
    return features

# Mantiene los valores ingresados ​​por el usuario en una variable
user_input = get_user_input()  # La variable user_input se utilizará para mostrar las entradas del usuario

# Creación de subtítulos para aplicaciones web y visualización de entradas de usuario
st.subheader('Entradas de usuario:')
st.write(user_input)

# Construye y entrena el modelo de Inteligencia Artificial
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Subtitula la aplicación web y ve las métricas del modelo (rendimiento)
st.subheader('Puntuación de precisión de la prueba del modelo de inteligencia artificial:')
# Prueba el modelo con el conjunto de datos Y_test y asigna el conjunto de datos X_test al modelo RandomForestClassifier, determinando la puntuación de precisión para predecir valores en Y_test
st.write('%' + str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100))  # multiplicado por 100 para obtener un porcentaje

# Asignar predicciones del modelo a una variable para determinar si el usuario cuya entrada (datos de salud) se toma es probable que tenga diabetes
prediction = RandomForestClassifier.predict(user_input)

# Crea subtítulos y muestra la aplicación web de clasificación (diabética o no)
st.subheader('Clasificación:')
st.write(prediction)
