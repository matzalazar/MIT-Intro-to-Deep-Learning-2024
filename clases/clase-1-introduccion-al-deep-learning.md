# Introducción al deep learning.

**¿Qué es la inteligencia?**

La habilidad de de procesar información para tomar decisiones en el futuro. Con esta definición, podemos pensar que la inteligencia artificial consiste en darle a las máquinas esa misma capacidad.

El machine learning es un subset de la inteligencia artificial. Es la ciencia que intenta enseñarle a las computadoras cómo hacer el procesamiento de la información y toma de decisiones a partir de los datos.

El deep learning es un subset del machine learning que usa redes neuronales para ello a partir de piezas de datos sin procesar.

**¿Por qué deep learning y por qué ahora?**

Los diseños manuales de algoritmos para determinar ciertos outpus demandan tiempo, son frágiles y en la práctica resultan muy poco escalables. ¿Podemos aprender las características suybacentes directamente desde los datos?

¿Cuáles son los patrones que debemos buscar en los datasets? 

Los algoritmos y la matemática subyacente al deep learning existen desde hace décadas. ¿Por qué lo estudiamos ahora?
1) **Big data**. Los datos están disponibles para nosotros está en todos lados. Son más abundantes ahora que lo que han sido nunca en nuestra historia.
2) **Hardware**. Estos algoritmos consumen una gran cantidad de recursos computacionales. Se benefician del hardware contemporáneo. 
3) **Software**. Hay programas open source que son usados como fundacionales para el deep learning.

## El perceptrón. Bloque estructural del deep learning.

Cada red neuronal está construida de múltiples perceptrones. 

Perceptrón: pensarlo como neurona individual. Una red neuronal está compuesta de muchas neuronas, y un perceptrón es sólo una neurona. 

### Propagación hacia adelante en una neurona:

![Forward Propagation](/recursos/imagenes/perceptron-forward-propagation-000.jpeg)

Una neurona ingiere información (muchas piezas de información). $X_1$, $X_2$, $X_m$. Cada uno de estos inputs será multiplicado elemento a elemento por un peso particular. $W_1$, $W_2$, $W_m$. Es decir, cada peso está asignado a un input. Finalmente, se suma el producto de cada input con su respectivo peso. Ese número final de la suma se pasa a través de una función no lineal de activación. Que producirá el output $y$.

$$
\hat{y} = g\left( \sum_{i=1}^m x_i w_i \right)
$$

Siendo $g$ la función no lineal de activación. Y $x_i, w_i$ cada input con su respectivo peso.

A esta estructura le falta lo que se conoce como bias (o término sesgado). Al que identificaremos como $w_0$.

$$
\hat{y} = g\left(w_0 + \sum_{i=1}^m x_i w_i \right)
$$

El término sesgado le permite a nuestra neurona modificar horizontalmente (en el plano de las $x$) la función no lineal.

Podemos reescribir esto para tratarlo de forma algebraica. 

$$
\hat{y} = g\left(w_0 + X^T W\right)
$$

donde $ X = \begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix} $ y $ W = \begin{bmatrix} w_1 \\ \vdots \\ w_m \end{bmatrix} $.

## Funciones de activación.

**Ejemplo de una función de activación**:

Si utilizamos una función de activación como Sigmoid, el resultado sería:

$$
g(z) = \sigma(z) = \frac{1}{1 + e^{-z}} 
$$

La función Sigmoid es muy utilizada ya que, dado cualquier input real, lo distribuye en un resultado (sobre el eje y) ubicado entre 0 y 1.

### Función Sigmoid.

$$
g(z) = \frac{1}{1 + e^{-z}}, \quad g'(z) = g(z)(1 - g(z))
$$

`tf.math.sigmoid(z)`

Es muy popular por su aplicación en la distribución probabilística. 

### Función Tanh.

$$
g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad g'(z) = 1 - g(z)^2
$$

`tf.math.tanh(z)`

### Función ReLU

$$
g(z) = \max(0, z), \quad g'(z) =
\begin{cases} 
1 & \text{si } z > 0 \\
0 & \text{en otro caso}
\end{cases}
$$

`tf.nn.relu(z)`

Es linear en todo su dominio excepto en $x=0$. 

![Funciones no lineales](/recursos/imagenes/common-activation-functions-000.jpeg)

### Importancia de las funciones no lineales de activación. 

Le permite a la red neuronal lidiar con información no lineal. Y de ese modo procesar datos que no son posibles de procesar a través de líneas.

![Funciones no lineales](/recursos/imagenes/importance-of-activation-functions-000.jpeg)

## Ejemplo de perceptrón.

![Ejemplo 1](/recursos/imagenes/perceptron-example-000.jpeg)

Tenemos $w_0 = 1$ y $W=\begin{bmatrix} 3 \\ -2 \end{bmatrix}$

Dos inputs: $x_1$ y $x_2$. Multiplicamos los nodos, sumamos el resultado y el sesgo. Y finalmente aplicamos la función no lineal. 

Ese es el proceso que se repite una y otra vez para cada neurona. 

Cada vez que eso sucede la neurona dará como output un único valor. 

$$
\hat{y} = g\left(w_0 + X^T W\right)
$$

$$
= g\left(1 + \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}^T\begin{bmatrix} 3 \\ -2 \end{bmatrix}\right)
$$

$$
= g(1 + 3x_1 - 2x_2)
$$

Donde $(1 + 3x_1 - 2x_2)$ es sólo una línea en dos dimensiones. Porque sólo tenemos dos parámetros en este modelo.

Si lo graficamos en un sistema de ejes cartesionamos, podremos ver exactamente cuál es el resultado que arroja esta neurona y qué espacios define.

Si a esa neurona le doy nueva data $\begin{bmatrix} -1 \\ 2 \end{bmatrix}$, la respuesta que la red nos dará sobre ese nuevo punto de dos parámetros dependerá de hacia qué lado del espacio caiga.

$$
\hat{y} = g\left(1 + (3*-1) - (2*2)\right)
$$

$$
\hat{y} = g(-6) \approx 0.002
$$

![Ejemplo 2](/recursos/imagenes/perceptron-example-2-000.jpeg)

La función Sigmoid divide el espacio en dos partes.

## Construir redes neuronales con perceptrones.

**El perceptrón simplificado**

![Simplificado](/recursos/imagenes/perceptron-simplified-000.jpeg)

Asumimos que cada línea está asociada a un peso específico, y que el término sesgado (bias) está incluido en el input.

$z$ es el la suma de los productos de peso e inputs, antes de pasar por la función no lineal.

De modo que el output final será $g(z)$: la función no lineal aplicada a $z$.

**Perceptrón con outputs múltiples**

![Multi output](/recursos/imagenes/multi-output-perceptron-000.jpeg)

Lo importante aquí es que cada neurona tiene sus pesos propios. 

Debido a que todas las entradas están densamente conectadas a todas las salidas, estas capas se denominan capas densas.

```python
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output
```

Aunque las librerías ya tienen las herramientas para implementar directamente:

```python
import tensorflow as tf

layer = tf.keras.layers.Dense(units=2)
```

Inicializamos una red de dos neuronas a través de una sola línea de código.

### Red neuronal de una sola capa.

Hay una capa entre nuestros inputs y nuestros outputs. Incrementamos la complejidad. Esta capa intermedia se llama capa invisible. (No es observable directamente, no pueda ser supervisada directamente).

Con una capa intermedia, tenemos también dos matrices de pesos. 

Cada capa oculta también tiene su función no lineal. 

![Red neuronal unicapa](/recursos/imagenes/single-layer-neural-network-000.jpeg)

El mismo proceso sucederá para todos los $z$, lo único que se modificará es el peso.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])
```

La función no lineal no necesariamente debe ser la misma en cada capa. Frecuentemente lo es, pero no es una condición indispensable. La que se aplica a cada neurona $z$ no tiene que ser obligatoriamente la misma que la que se aplica en el output final de esta red, por ejemplo.

### Red neuronal profunda.

![Red profunda](/recursos/imagenes/deep-neural-network-000.jpeg)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n₁),
    tf.keras.layers.Dense(n₂),
    .
    .
    .
    tf.keras.layers.Dense(2)
])
```

## Aplicaciones de redes neuronales.

**Ejemplo**

Sistema de IA que determinará si un estudiante aprobará o no una cátedra.

Modelo de dos inputs: 
- $x_1 =$ número de clases a las que asistió.
- $x_2 =$ horas dedicadas al proyecto final.

Recuperamos valores para este sistema de años anteriores. 

Debemos indicarle a la red cuándo está tomando malas decisiones. Eso se hace alimentándola con resultados de años anteriores. De esta manera, generamos una red que minimiza la pérdida entre las predicciones y los outputs.

```python
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y, predicted) )
```

Con la función softmax_cross podemos reducir la pérdida para outputs esperables de 0 o 1. Es decir, valores booleanos de verdader o falso. Aprobará / No aprobará.

En cambio, si quisiéramos saber la nota que vamos a sacarnos:

```python
loss = tf.reduce_mean( tf.square(tf.subtract(y, predicted)) )
loss = tf.keras.losses.MSE(y, predicted)
```

## Entrenar redes neuronales.

Queremos encontrar los pesos de nuestra red neuronal que minimicen la pérdida. (Encontrar el vector $W$ óptimo según toda la data de la que se dispone).

Recordemos que $W$ es una matriz de números (una lista de pesos) para cada una de las capas y cada una de las neuronas. Y cada capa tendrá asociada una matriz específica para sus pesos.

¿Cómo lo hacemos? Empezamos con un lugar aleatorio en el espacio. Y medimos en el espacio niveles cada vez más bajos del gradiente. Repetimos ese paso hasta que encontramos un mínimo local. 

**Descenso de gradiente**

Algoritmo:
1) Inicializar pesos de forma aleatoria. $\sim \mathcal{N}(0, \sigma^2)$
2) Iterar hasta la convergencia. 
3) Calcular el gradiente. $\frac{\partial J(W)}{\partial W}$
4) Actualizar los pesos. $W\leftarrow W - \eta \frac{\partial J(W)}{\partial W}$
5) Retornar los pesos.

En código:

```python
import tensorflow as tf

weights = tf.Variable([tf.random.normal()])

while True:     # loop forever
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)

    weights = weights - lr * gradient
```

LR es "learning rate". $\eta$

El término gradient nos indica cuándo ascendemos en el espacio. No sólo eso. También nos dice cómo es el espacio, cómo nuestra pérdida va modificándose.

**Retropropagación**

¿Cómo un pequeño cambio en uno de los pesos afecta la pérdida final?

$\frac{\partial J(W)}{\partial w_2} = \frac{\partial J(W)}{\partial \hat y} * \frac{\partial \hat y}{\partial w_2}$

Esto se computa una y otra vez desde los outputs hasta los inputs.

## Redes neuronales en la práctica. Optimización.

El algoritmo de retropropagación, si bien es fácil de entender, demanda muchísimos recursos computacionales.

Las funciones de pérdida pueden ser muy difíciles de optimizar. Entonces, ¿cómo podemos setear el parámetro de la tasa de aprendizaje?

En: $W\leftarrow W - \eta \frac{\partial J(W)}{\partial W}$, el término $\eta$.

¿Cómo setearlo correctamente en la práctica?
1) Probar diferentes $\eta$ y ver cómo funcionan. 
2) O también, algo más inteligente, es diseñar un $\eta$ adaptativo que se amolde al espacio.
- Ya no son fijos. Se agradarán o achicarán según el largo del gradiente, qué tan rápido entiende qué está sucediendo, el tamaño de pesos particulares, etc.

**Algoritmos**
1) SGD. `tf.keras.optimizers.SGD`
2) Adam. `tf.keras.optimizers.Adam`
3) Adadelta. `tf.keras.optimizers.Adadelta`
4) Adagrad. `tf.keras.optimizers.Adagrad`
5) RMSProp. `tf.keras.optimizers.RMSProp`

### Definir el modelo.

```python
import tensorflow as tf

model = tf.keras.Sequential([...])

# pick your favorite optimizer
optimizer = tf.keras.optimizer.SGD()

while True: # loop forever

    # forward pass trough the network
    prediction = model(x)

    with tf.GradientTape() as tape:
        # compute the loss
        loss = compute_loss(y, prediction)
    
    # update the weights using the gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## Redes neuronales en la práctica. Mini lotes.

Por lo general, se calculan los gradientes usando lo que se conoce como mini batch (un mini lote). No sólo un input determinado (lo que sería demasiado sesgado) ni tampoco todo el lote de data (lo que sería computacionalmente demandante), sino una parte del mismo.

En general, se utilizan 32 piezas de información. Eso nos da un estimado del verdadero gradiente. 

Entonces, los mini lotes permiten: una estimación más exacta del gradiente y una convergencia suave. Además, permiten paralelizar el cómputo. Lo que hace que el proceso sea mucho más veloz. 

## Redes neuronales en la práctica. Sobreajuste.

El sobreajuste es un problema que existe no sólo en el deep learning sino en toda la rama del machine learning. 

¿Cómo definir si el modelo está reflejando todos los datos?

![Overfitting](/recursos/imagenes/the-problem-of-overfitting-000.jpeg)

Básicamente, es un problema de complejidad: un modelo muy simple no logrará evaluar parámetros relevantes, y un modelo muy complejo tomará parámetros excesivos innecesariamente.

Si nuestro modelo funciona muy bien en el dataset de entrenamiento pero muy mal en el dataset de testeo, eso significa que está sobreajustado. 

El otro problema es que el modelo sea subajustado. Es decir, que no tenga suficiente data de la cual alimentarse.

### Regularización.

Técnica que limita nuestro problema de optimización para no generar modelos complejos.

Mejora nuestro modelo para los datasets de testeo. 

1) Dropout. Durante el entrenamiento, aleatoriamente seteamos algunas activaciones a 0.
Típicamente, hacemos que el 50% de las activaciones en una capa sea 0.
`tf.keras.layers.Dropout(p=0.5)`
En cada iteración, los nodos que se desactivación serán distintos.

2) Detenerlo tempranamente.
Es decir, detener el entrenamiento antes de tener una chance de sobreajuste.

