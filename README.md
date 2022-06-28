# Documentación técnica TFG

Introducción

El proyecto ha consistido en el desarrollo de un sistema de mineria de opiniones con datos de la red social Twitter, con el objetivo de permitir la identificacion del sentimiento de odio y la detección de ciertos sesgos sociales como la discriminación racial y de genero. Para ello, tras el estudio de los requisitos y el diseño del sistema, el producto final se encuentra estructurado en tres capas. 

La primera, la capa de ingesta, sirve como punto de entrada a la aplicacion, y cumple la funcion de conformar el Corpus que servira de entrada al sistema de mineria de opiniones. Bien puede tratarse de los ficheros ya cargados en la aplicacion o de aquellos que el usuario quiera importar directamente.

La segunda capa es la capa de visualización, se encarga de visualizar todo el desarrollo que se lleva a cabo en la Capa de minería de opiniones. Se trata de un Dashboard creado con la herramienta de Streamlit. En ella, se visualizará todo el proceso de analisis de sentimientos, la gestion de los parametros llevados a cabo con el usuario y la traduccion de la aplicacion. 

La tercera, que corresponde a la Capa de mineria de opiniones, consta de un consumidor de texto; que leera los datasets elegidos, un preprocesado de los mismos, el entrenamiento del clasificador Naive Bayes y por ultimo la validacion del algoritmo calculando las metricas y obteniendo la matriz de confusion.

Descripción del repositorio

