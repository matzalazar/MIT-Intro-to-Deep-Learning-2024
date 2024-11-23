# Redes neuronales recurrentes.

Problemas en modelos secuenciales. Datos secuenciales o procesamiento secuencial de datos.

_Por ejemplo_: dada la imagen de una pelota, ¿podemos predecir cuál será su siguiente movimiento? Sin información previa sobre la trayectoria, resultaría una predicción demasiado aleatoria o caprichosa. Pero, si además de la posición actual de la pelota, agregamos dónde estaba la pelota previamente, el problema se convierte en algo más simple. Y entonces sí podemos hacer una predicción bastante acertada.

**Esto es información secuencial**.

La información secuencial nos rodea: nuestras voces pueden ser pensadas como pedazos de información secuencial en forma de ondas sonoras, nuestro lenguaje en forma de palabras, etc. 

## Aplicaciones del modelado secuencial.

Con esta estructura secuencial, podemos tomar una oración y analizar si refleja, por ejemplo, un sentimiento positivo (input secuencial, output único). También podemos generar secuencias a partir de otros datos (tenemos una imagen y queremos describirla: dado un input - la imagen - generamos un output secuencial). O también podemos tener secuencias como inputs y secuencias como outpus en, por ejemplo, un traductor.

!! Dispositiva 8.

## Neuronas con recurrencia.

En el modelo de neurona básico (múltiples inputs que se combinan linealmente con el vector de pesos específicos cuya suma será atravesada por una función no lineal que generará un output) aún no tenemos una noción real de secuencia o tiempo.

**Información secuencial es información a través del tiempo.**

Podemos aplicar un modelo paso a paso en cada momento t de tiempo de la secuencia.

La idea clave es asociar el cómputo entre los distintos momentos t de procesamiento. 

Podemos hacerlo matemáticamente introduciendo una variable que llamaremos $h$. De modo que $h_t$ refiere a esta noción de estado de la red neuronal. Ese estado es aprendido y computado por la neurona en esta capa. Y es pasada y propagada en cada paso de tiempo.

!! Dispositiva 15

Generamos una relación en la que el output en un instante $t$ ahora depende simultáneamente de un input en ese instante $t$ y del estado anterior que acaba de ser procesado.

$$
\hat{y} = f(x_t, h_{t-1})
$$

Es decir, pasamos el estado de la red hacia adelante a lo largo del tiempo. Lo que constituye la base de la **recurrencia en las neuronas**.

El output es un producto del input presente y la memoria previa. 

El estado oculto $h_t$ actúa como una memoria que codifica información relevante tanto del input actual $x_t$ como de los estados previos $h_{t-1}$.

## Redes neuronales recurrentes. RNN.

Las RNR son la base para resolver problemas de modelado secuencial.

La idea clave es introducir la variable $h_t$, que se actualiza a cada paso de $t$ a medida que procesamos la secuencia. Esa actualización es almacenada en lo que se conoce como una relación recurrente.

$$
h_t = f_w(x_t, h_{t-1})
$$

!! Dispositiva 17

### Implementación.

En la implementación, `hidden_state` se pasa entre los pasos para mantener la memoria, y `my_rnn` reutiliza los mismos pesos en cada iteración para garantizar consistencia."

```python
# Definición e implementación básica de una RNN
my_rnn = RNN()  # Modelo predefinido de red neuronal recurrente
hidden_state = [0, 0, 0, 0]  # Estado oculto inicial (vector inicializado en ceros)

# Frase de entrada como una lista de palabras
sentence = ["I", "love", "recurrent", "neural"]

# Procesamos cada palabra secuencialmente
for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

# Predicción de la siguiente palabra tras procesar toda la frase
next_word_prediction = prediction
print(next_word_prediction)  # >>> "networks"
```

!! Dispositiva 20

!! Dispositiva 24

La idea es poder visualizar el output que va produciéndose en cada paso de la secuencia de $t$. Reusamos la misma matriz de pesos en cada paso.

!! Dispositiva 26

Así obtenemos un índice de pérdida (loss) y a través de técnicas de retropropagación entender cómo ajustar nuestros pesos. Y ese proceso lo hacemos en cada nivel de tiempo. De modo que finalmente conocemos el índice de pérdida total $L$ correspondiente a toda la secuencia.

### Implementación.

```python
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        # Inicializar matriz de pesos
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        # Inicializar estados ocultos en ceros
        self.h = tf.zeros([rnn_units, 1])
    
    def call(self, x):
        # Actualizar el estado oculto
        self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)

        # Computar la salida
        output = self.W_hy * self.h

        # Retornar el output y el estado oculto actuales
        return output, self.h
```

Tensorflow lo simplifica con funciones nativas:

`tf.keras.layers.SimpleRNN(rnn_units)`

## Diferente tipos de problemas que el modelado secuencial de RNN permite resolver.

1. Muchos a uno. Ejemplo: análisis de sentimientos. 
   - Se reciben múltiples inputs, procesados secuencialmente, y el resultado es un único output (sentimiento positivo / sentimiento negativo).
2. Uno a mucho. Ejemplo: descripción de imágenes.
    - Se recibe un único input (la imagen), y procesada secuencialmente, se generan outputs (palabras) que la describen.
3. Muchos a muchos. Ejemplo: traductor.
    - Se reciben múltiples inputs (palabras) de forma secuencial, y se generan múltiples outputs (palabras) de forma secuencial.

## Modelado secuencial. Criterios para su diseño.

Necesitamos:
- **Manejar secuencias de longitud variable**. Algunas oraciones pueden tener cinco palabras, otras pueden tener cien palabras. Necesitamos flexibilidad en nuestro modelo para poder gestionar estos casos.
- **Seguimiento de dependencias a largo plazo**. Mantener una especie de memoria, hacer un seguimiento de las dependencias que ocurren en las secuencias. Cosas que aparecen en momentos muy tempranos pueden tener una importancia relevante más adelante, etc. 
- **Información sobre el orden**. Las secuencias poseen inherentemente un orden. Necesitamos preservalo.
- **Compartir parámetros a lo largo de la secuencia**. 

Las RNN nos dan la posibilidad de hacer todas estas cosas. 

## Ejemplo de modelado de un problema secuencial: predecir la siguiente palabra.

"Esta mañana llevé a mi gato a pasear".

Dadas las primeras palabras de la oración, la tarea consiste en predecir la última palabra.

1. Debemos representar el lenguaje. Considerando que las redes neuronales están compuestas de operadores numéricos. Hay que encontrar una forma de representar la palabra de modo numérico. 

¿Cómo codificar lenguaje de manera que sea entendible para una red neuronal?

Idea de incrustación (embedding). Transformar inputs en vectores numéricos de un determinado tamaño para así poder operar con ellos.

!! Dispositiva 36

Podemos asignar un índice a cada palabra. Una manera muy simple y muy poderosa. Otra idea es distribuir las palabras en un espacio vectorial, de modo que las palabras que se relacionan en el lenguaje estén cercanas en ese espacio. Y cosas que sean diferentes en el lenguaje sea, a su vez, numéricamente diferentes. 

2. Debemos manejar secuencias de distinto tamaño. Oraciones de cuatro palabras, oraciones de seis palabras, etc. La red debería ser capaz de manejar esas variaciones. 

A su vez, debería recuperar información de posiciones distantes en la secuencia. (Esta es la idea de "memoria" de la red). 

3. Capturar diferencias en el orden de la secuencia.

"La comida estaba rica, no fea" / "La comida estaba fea, no rica".

Incluso si se trata del mismo grupo de palabras, el orden de las mismas influye en el significado. 

**Criterios**:
- Manejar secuencias de distinta longitud.
- Trackear dependencias a largo plazo.
- Mantener información sobre el orden.
- Compartir parámetros a lo largo de la secuencia.

## Retropropagación a lo largo del tiempo. BPTT.

Cuando retrotraemos el output para recalcular los pesos y minimzar la pérdida.

MINUTO 33.