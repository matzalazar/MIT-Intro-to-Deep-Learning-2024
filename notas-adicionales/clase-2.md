# RNN State Update and Output

## Componentes clave

1. **Input Vector ($x_t$):**
   - Este es el dato de entrada en el momento actual \(t\). Por ejemplo, podría ser una palabra, un pixel, o cualquier característica de un dato secuencial.

2. **Hidden State ($h_t$):**
   - Es el "estado oculto" en el momento actual \(t\), que captura información tanto de la entrada actual (\(x_t\)) como del estado oculto previo (\(h_{t-1}\)).
   - Este estado es esencial para mantener la dependencia temporal, ya que guarda información del contexto anterior.

3. **Output Vector ($\hat{y}_t$):**
   - Este es el valor de salida en el momento \(t\). Puede representar, por ejemplo, una predicción, una clasificación, o cualquier otra salida según el problema.

## Fórmulas

### Actualización del estado oculto ($h_t$):
$$
h_t = \tanh(W_{hh}^T h_{t-1} + W_{xh}^T x_t)
$$
- **Explicación:**
  - $W_{hh}^T$: Matriz de pesos que relaciona el estado oculto previo $h_{t-1}$ con el actual.
  - $W_{xh}^T$: Matriz de pesos que relaciona la entrada actual $x_t$ con el estado oculto actual $h_t$.
  - $tanh$: Función de activación no lineal que asegura que el estado oculto esté en un rango limitado y permite modelar relaciones complejas.
  - El estado oculto $h_t$ combina información del pasado $h_{t-1}$ con la entrada actual $x_t$.

### Generación del vector de salida $\hat{y}_t$:
$$
\hat{y}_t = W_{hy}^T h_t
$$
- **Explicación:**
  - $W_{hy}^T$: Matriz de pesos que transforma el estado oculto actual $h_t$ en la salida deseada $\hat{y}_t$.
  - Esto conecta el estado interno con lo que la red produce como salida, que puede ser una predicción o cualquier valor relevante.

## Flujo de datos

1. En el momento $t$, la red toma:
   - El vector de entrada actual $x_t$.
   - El estado oculto previo $h_{t-1}$.

2. Calcula el nuevo estado oculto $h_t$ usando la ecuación de actualización.

3. Usa $h_t$ para calcular la salida $\hat{y}_t$.

4. Este proceso se repite para todos los pasos de tiempo $t$ de la secuencia, permitiendo que la red aprenda dependencias temporales.

## Interpretación

- **Dependencias temporales:** La RNN utiliza $h_{t-1}$ para recordar información de pasos previos, lo que la hace ideal para datos secuenciales como texto, audio o series temporales.
- **Flexibilidad:** La misma fórmula se aplica a cada paso $t$, permitiendo manejar secuencias de longitud variable.
- **Limitación:** El uso de $tanh$ puede provocar problemas como el desvanecimiento del gradiente, dificultando el aprendizaje de dependencias a largo plazo (problema que solucionan arquitecturas como LSTM o GRU).
