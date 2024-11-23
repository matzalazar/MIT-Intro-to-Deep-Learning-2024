import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generar datos secuenciales simples: 1, 2, 3 -> 4, 2, 3, 4 -> 5, etc.
# Vamos a generar secuencias de longitud 3 como entrada y 1 como salida.

# Datos de entrada (X) y salida (y)
def generate_data():
    X, y = [], []
    for i in range(1, 100):
        X.append([i, i+1, i+2])  # Secuencia de entrada
        y.append(i+3)  # Siguiente número en la secuencia
    return np.array(X), np.array(y)

X, y = generate_data()

# Redimensionar los datos para que sean compatibles con la entrada de la RNN
# Keras requiere la forma (muestras, pasos de tiempo, características)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Crear el modelo
model = Sequential([
    SimpleRNN(20, activation='relu', input_shape=(X.shape[1], X.shape[2])),  # 10 unidades recurrentes
    Dense(1)  # Capa de salida con un solo valor
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X, y, epochs=20, batch_size=10)

# Probar el modelo
test_input = np.array([[101, 102, 103]])  # Nueva secuencia para predecir
test_input = test_input.reshape((1, 3, 1))
prediction = model.predict(test_input)

print("Predicción para la secuencia [101, 102, 103]:", prediction)
