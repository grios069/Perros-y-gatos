# Esta línea importa la biblioteca TensorFlow bajo el alias tf. 
import tensorflow as tf
# En esta boblioteca TFDS es una biblioteca que proporciona una colección de conjuntos de datos listos para usar en TensorFlow.
import tensorflow_datasets as tfds

#Se establece una nueva URL para descargar el conjunto de datos de perros y gatos, reemplazando la URL existente.
setattr(tfds.image_classification.cats_vs_dogs, '_URL', "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

# Descargar el conjunto de datos de perros y gatos
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)



#Imprimir los metadatos para revisarlos
metadatos


#Una forma de mostrar 5 ejemplos del set
tfds.as_dataframe(datos['train'].take(5), metadatos)


#Otra forma de mostrar ejemplos del set
tfds.show_examples(datos['train'], metadatos)

# Manipular y visualizar el set
# Lo pasamos a TAMANO_IMG (100x100) y a blanco y negro (solo para visualizar)
import matplotlib.pyplot as plt
import cv2

# Crear una figura de tamaño 20x20 pulgadas para visualizar las imágenes
plt.figure(figsize=(20, 20))

# Definir el tamaño deseado para las imágenes
TAMANO_IMG = 100

# Iterar sobre los primeros 25 datos de entrenamiento
for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
    # Cambiar el tamaño de la imagen al tamaño deseado utilizando cv2.resize()
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # Convertir la imagen a escala de grises utilizando cv2.cvtColor()
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Crear un subplot en la figura de 5x5 y mostrar la imagen en escala de grises
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap='gray')
    


#Variable que contendra todos los pares de los datos (imagen y etiqueta) ya modificados (blanco y negro, 100x100)
datos_entrenamiento = []


for i, (imagen, etiqueta) in enumerate(datos['train']):  # Todos los datos
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG)) #utilizando la función cv2.resize(). imagen.numpy() convierte la imagen en un arreglo NumPy antes de redimensionarla.
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)  # Cambiar tamaño a 100x100x1
    datos_entrenamiento.append([imagen, etiqueta])


#------------------------------------------------------------------------------------------------

#Ver los datos del primer indice
datos_entrenamiento[0]


#Ver cuantos datos tengo en la variable
len(datos_entrenamiento)


# Preparar mis variables X (entradas) y y (etiquetas) separadas

X = []  # Imágenes de entrada (píxeles)
y = []  # Etiquetas (perro o gato)

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen) #Se acumulan todas las imágenes de entrada en la lista X.
    y.append(etiqueta)# Se acumulan todas las etiquetas correspondientes a las imágenes en la lista y.

#--------------------------------------------------------------------------------------------------------
#Normalizar los datos de las X (imagenes). Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
import numpy as np

X = np.array(X).astype(float) / 255
y

#----------------------------------------------------------------------------------------

#Convertir etiquetas en arreglo simple
y = np.array(y)

#se obtiene una tupla que indica la cantidad de imágenes en el arreglo X y las dimensiones espaciales de cada imagen.
X.shape

#Crear los modelos iniciales
#Usan sigmoid como salida (en lugar de softmax) para mostrar como podria funcionar con dicha funcion de activacion.
#Sigmoid regresa siempre datos entre 0 y 1. Realizamos el entrenamiento para al final considerar que si la respuesta se
#acerca a 0, es un gato, y si se acerca a 1, es un perro.

modeloDenso = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN2 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


#Compilar modelos. Usar crossentropy binario ya que tenemos solo 2 opciones (perro o gato)
modeloDenso.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloCNN.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloCNN2.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])




#La variable de tensorboard se envia en el arreglo de "callbacks" (hay otros tipos de callbacks soportados)
#En este caso guarda datos en la carpeta indicada en cada epoca, de manera que despues
#Tensorboard los lee para hacer graficas
tensorboardDenso = TensorBoard(log_dir='logs/denso')
modeloDenso.fit(X, y, batch_size=32,
                validation_split=0.15,
                epochs=100,
                callbacks=[tensorboardDenso])



#Cargar la extension de tensorboard de colab
%load_ext tensorboard



#Ejecutar tensorboard e indicarle que lea la carpeta "logs"
%tensorboard --logdir logs

#este código utiliza TensorBoard para realizar el seguimiento y registro de métricas durante el entrenamiento del modelo modeloCNN. 
tensorboardCNN = TensorBoard(log_dir='logs/cnn')
modeloCNN.fit(X, y, batch_size=32,
                validation_split=0.15,
                epochs=100,
                callbacks=[tensorboardCNN])


tensorboardCNN2 = TensorBoard(log_dir='logs/cnn2')
modeloCNN2.fit(X, y, batch_size=32,
                validation_split=0.15,
                epochs=100,
                callbacks=[tensorboardCNN2])


#ver las imagenes de la variable X sin modificaciones por aumento de datos
plt.figure(figsize=(20, 8))
for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(X[i].reshape(100, 100), cmap="gray")
  
  
  #Realizar el aumento de datos con varias transformaciones. Al final, graficar 10 como ejemplo
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X)

plt.figure(figsize=(20,8))

for imagen, etiqueta in datagen.flow(X, y, batch_size=10, shuffle=False):
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i].reshape(100, 100), cmap="gray")
  break


#Este modelo es un modelo denso (totalmente conectado) con tres capas ocultas. 
# La entrada se aplana utilizando la capa Flatten,
# que convierte la imagen de entrada en un vector unidimensional. 
# Luego, se agregan dos capas densas con activación ReLU (Dense) con 150 neuronas cada una. 
# Finalmente, se agrega una capa densa de salida con activación sigmoide para clasificación binaria.

modeloDenso_AD = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#Este modelo es una red neuronal convolucional (CNN) con capas convolucionales y capas de agrupación. 
# Se utilizan varias capas Conv2D seguidas de capas MaxPooling2D para extraer características y reducir la dimensionalidad. 
modeloCNN_AD = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#Este modelo es similar al anterior, pero con una capa adicional de dropout (Dropout) antes de la capa densa. 
# La capa de dropout ayuda a evitar el sobreajuste al desactivar aleatoriamente algunas neuronas durante el entrenamiento. 
modeloCNN2_AD = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#Define el optimizador, la función de pérdida y las métricas a utilizar durante el entrenamiento y la evaluación de los modelos.
modeloDenso_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloCNN_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloCNN2_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])



#Separar los datos de entrenamiento y los datos de pruebas en variables diferentes

len(X) * .85 #19700
len(X) - 19700 #3562

X_entrenamiento = X[:19700]
X_validacion = X[19700:]

y_entrenamiento = y[:19700]
y_validacion = y[19700:]



#Usar la funcion flow del generador para crear un iterador que podamos enviar como entrenamiento a la funcion FIT del modelo
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)


#el código configura y realiza el entrenamiento del modelo modeloDenso_AD utilizando un generador de datos,
# un número específico de épocas y tamaño de lote, datos de validación, y registra los eventos del entrenamiento utilizando TensorBoard.
tensorboardDenso_AD = TensorBoard(log_dir='logs/denso_AD')

modeloDenso_AD.fit(
    data_gen_entrenamiento,
    epochs=100, batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32))),
    callbacks=[tensorboardDenso_AD]
)

tensorboardCNN_AD = TensorBoard(log_dir='logs-new/cnn_AD')

modeloCNN_AD.fit(
    data_gen_entrenamiento,
    epochs=150, batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32))),
    callbacks=[tensorboardCNN_AD]
)

tensorboardCNN2_AD = TensorBoard(log_dir='logs/cnn2_AD')

modeloCNN2_AD.fit(
    data_gen_entrenamiento,
    epochs=100, batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32))),
    callbacks=[tensorboardCNN2_AD]
)

#guarda el modelo modeloCNN_AD en un archivo con formato h5. 
modeloCNN_AD.save('perros-gatos-cnn-ad.h5')

#se utiliza en un entorno de Python para instalar el paquete tensorflowjs. Este paquete proporciona herramientas y utilidades para convertir modelos de TensorFlow en formatos compatibles con TensorFlow.js
!pip install tensorflowjs

#se utiliza para crear directorios en sistemas operativos
!mkdir carpeta_salida

#se utiliza para convertir un modelo de TensorFlow guardado en formato h5 (formato utilizado por Keras) en un formato compatible con TensorFlow.js.
!tensorflowjs_converter --input_format keras perros-gatos-cnn-ad.h5 carpeta_salida