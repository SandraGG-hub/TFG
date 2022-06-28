# Documentación técnica TFG

## Introducción

El proyecto ha consistido en el desarrollo de un sistema de minería de opiniones con datos de la red social Twitter, con el objetivo de permitir la identificación del sentimiento de odio y la detección de ciertos sesgos sociales como la discriminación racial y de género. Cabe destacar que se han llevado a cabo dos fases: por un lado, se ha realizado un análisis para identificar la mejor herramienta de cara al idioma español, así como utilizando ciertas variantes de configuración en los parámetros para realizar una comparativa; y a mayores la creación de la aplicación web. Para ello, tras el estudio de los requisitos y el diseño del sistema, el producto final se encuentra estructurado en tres capas. 

La primera, la capa de ingesta, sirve como punto de entrada a la aplicación, y cumple la función de conformar el Corpus que servira de entrada al sistema de minería de opiniones. Bien puede tratarse de los ficheros ya predefinidos en la aplicación o de aquellos que el usuario quiera importar directamente.

La segunda capa es la capa de visualización, se encarga de visualizar todo el desarrollo que se lleva a cabo en la Capa de minería de opiniones. Se trata de un Dashboard creado con la herramienta de Streamlit. En ella, se visualizará todo el proceso de analisis de sentimientos, la gestion de los parametros llevados a cabo con el usuario y la traduccion de la aplicacion. 

La tercera, que corresponde a la Capa de minería de opiniones, consta de un consumidor de texto; que leera los datasets elegidos, un preprocesado de los mismos, el entrenamiento del clasificador Naive Bayes y por ultimo la validación del algoritmo calculando las métricas y obteniendo la matriz de confusión.

## Descripción del repositorio

### Dashboard
Se corresponde al fichero que recibe el nombre de "myapp.py", aquí se contiene todo el código fuente correspondiente a la aplicación que se visualiza en Streamlit y que podemos encontrar accediendo a esta URL: https://sandragg-hub-tfg-myapp-2f2rtv.streamlitapp.com/ 

### Locales
Se corresponde con la carpeta que contiene toda la información necesaria para traducir la aplicación tanto al idioma inglés como al idioma español. Estos ficheros son los que se crean haciendo uso del módulo PyGettext de python. Es por ello, que dentro de ella podemos encontrar a su vez dos carpetas llamadas "es" y "en" que corresponden a cada uno de los idiomas mencionados. 

### Requirements
En este archivo de texto se vuelca toda la información correspondiente a las diferentes versiones de las librerías utilizadas en el desarrollo. Es necesario para que a la hora de realizar el despliegue de la aplicación en Streamlit, no haya problemas debido a incompatibilidades.

### Auxiliar
En esta carpeta se encuentran todos los ficheros auxiliar de los que hará uso la aplicación. Tanto de los archivos correspondientes a los datasets cargados, como los archivos correspondientes a la herramienta TreeTagger. Cabe destacar que algunos ficheros no se utilizan para la aplicación desplegada ya que corresponden al análisis realizado previamente. 



