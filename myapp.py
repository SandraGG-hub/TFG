#importamos librerías
#!/bin/sh
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import treetaggerwrapper
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from PIL import Image
from sklearn.metrics import classification_report
import gettext



#método auxiliar para comprobar si dos listas son iguales
def iguales(lista1, lista2):
    for i in range(len(lista1)):
        if lista1[i]!=lista2[i]:
            return False
    return True

#método para descarga de CSV 
def filedownload(df, language, final):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        #opción idioma español
        if language == 'es' and final == 0:
            href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Descargar CSV</a>'
        #opción idioma inglés
        elif language == 'en' and final == 0:
            href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV</a>'
        #opción idioma español y obtener resultado
        if language == 'es' and final == 1:
            href = f'<a href="data:file/csv;base64,{b64}" download="resultado.csv">Descargar CSV</a>'
        #opción idioma inglés t obtener resultado
        elif language == 'en' and final == 1:
            href = f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download CSV</a>'
        return href

#sistema de minería de opiniones con herramienta NLTK
def analisis_1(dataset_clasificador, max_feat, ngram_1, ngram_2, class_names, test, train, language):
        
        #leer dataset
        tweet = pd.read_csv(dataset_clasificador, header=None, sep=';', names=['label', 'tweets'])

        #etiquetado de sentimientos
        if dataset_clasificador == 'traducir/total_tuits_odio.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1})
        elif dataset_clasificador == 'traducir/total_tuits_odio_1discriminacion.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2})
        elif dataset_clasificador == 'traducir/total_tuits_odio_2discriminaciones.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2, 'discriminacion genero':3})
        else:
                st.write(f'{dataset_clasificador}')
                if len(class_names) == 2:
                        tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1})
                elif len(class_names) == 3:
                        tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2})
                elif len(class_names) == 4:
                         tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2, 'discriminacion genero':3})
                         
        #simplificación de carcajadas en caso de que las hubiera
        tweet.replace('jajaja', 'jaja')
        tweet.replace('jajajaj', 'jaja')
        tweet.replace('jajajaja', 'jaja')
        tweet.replace('jajajajajaj', 'jaja')
        tweet.replace('jajajajajjaja', 'jaja')
        tweet.replace('jajaj', 'jaja')
        tweet.replace('jaaja', 'jaja')
        tweet.replace('jaaaja', 'jaja')
        tweet.replace('jajajajaja', 'jaja')
        tweet.replace('jajajajajajaja', 'jaja')
        tweet.replace('jajajajajajajajaja', 'jaja')
        tweet.replace('jajajajajajajajajaja', 'jaja')

        #preprocesamiento
        preproceso=[]
        stopWords = set(stopwords.words('spanish'))
        stemmer = SnowballStemmer('spanish') 

        #tokenización
        for x in tweet['tweets']:
            words = word_tokenize(x)
            wordsFiltered = []
            for w in words:
                if w not in stopWords:
                    wordsFiltered.append(stemmer.stem(w)) 
            lista_tokens =" ".join(wordsFiltered)
            preproceso.append(lista_tokens)
            
        #ajuste idioma español
        if language == 'es':
            tweet['preproceso']=preproceso
        #ajuste idioma inglés
        elif language == 'en':
            tweet['preprocess']=preproceso
        st.subheader(_('Preprocesamiento'))

        if language == 'es':
            st.write(tweet['preproceso'])
        elif language == 'en':
            st.write(tweet['preprocess'])

        if language == 'es':
            x= tweet.preproceso
        if language == 'en':
            x= tweet.preprocess
        y= tweet.label_num

        #división del conjunto de datos
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=test, train_size=train)
        
        vect = CountVectorizer(max_df=0.5, min_df=2, ngram_range=(ngram_1, ngram_2), max_features = max_feat)
        x_train_dtm = vect.fit_transform(x_train)
        x_test_dtm = vect.transform(x_test)
        st.subheader(_('División del conjunto de datos'))
        st.markdown(_('Set de entrenamiento (80%):'))
        st.write(x_train_dtm.shape)
        st.markdown(_('Set de prueba (20%):'))
        st.write(x_test_dtm.shape)

        #creación bolsa de palabras
        st.subheader(_('Bolsa de palabras'))
        matriz= pd.DataFrame(x_train_dtm.toarray(), columns=vect.get_feature_names())
        st.write(matriz)
        
        #entrenamiento del clasificador
        nb_multi = MultinomialNB()
        nb_multir = nb_multi.fit(x_train_dtm, y_train)

        y_pred_class = nb_multi.predict(x_test_dtm)

        #cálculo de métricas
        st.subheader(_('Métricas'))
        st.markdown(_('Accuracy:'))
        st.info(accuracy_score(y_test, y_pred_class))
        st.markdown(_('Precision:'))
        st.info(precision_score(y_test, y_pred_class, average='weighted'))
        st.markdown(_('Recall:'))
        st.info(recall_score(y_test, y_pred_class, average='weighted'))
        st.markdown(_('F1-score:'))
        st.info(f1_score(y_test, y_pred_class, average='weighted'))
        

        #crear matriz de confusión
        matrix = confusion_matrix(y_test, y_pred_class)

        #crear marco de datos de pandas 
        dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

        return dataframe

#sistema de minería de opiniones con herramienta TreeTagger
def analisis_2(dataset_clasificador, max_feat, ngram_1, ngram_2, class_names, test, train, language):

        #leer dataset
        tweet = pd.read_csv(dataset_clasificador, header=None, sep=';', names=['label', 'tweets'])

        #etiquetado de sentimientos    
        if dataset_clasificador == 'traducir/total_tuits_odio.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1})
        elif dataset_clasificador == 'traducir/total_tuits_odio_1discriminacion.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2})
        elif dataset_clasificador == 'traducir/total_tuits_odio_2discriminaciones.csv':
            tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2, 'discriminacion genero':3})
        else:
                st.write(f'{dataset_clasificador}')
                if len(class_names) == 2:
                        tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1})
                elif len(class_names) == 3:
                        tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2})
                elif len(class_names) == 4:
                         tweet['label_num']=tweet.label.map({'odio':0, 'no odio':1, 'discriminacion racial':2, 'discriminacion genero':3})
                         
        #simplificación de carcajadas en caso de que las hubiera
        tweet.replace('jajaja', 'jaja')
        tweet.replace('jajajaj', 'jaja')
        tweet.replace('jajajaja', 'jaja')
        tweet.replace('jajajajajaj', 'jaja')
        tweet.replace('jajajajajjaja', 'jaja')
        tweet.replace('jajaj', 'jaja')
        tweet.replace('jaaja', 'jaja')
        tweet.replace('jaaaja', 'jaja')
        tweet.replace('jajajajaja', 'jaja')
        tweet.replace('jajajajajajaja', 'jaja')
        tweet.replace('jajajajajajajajaja', 'jaja')
        tweet.replace('jajajajajajajajajaja', 'jaja')

        #preprocesamiento
        preproceso=[]
        tagger=treetaggerwrapper.TreeTagger(TAGLANG='es', TAGDIR='traducir/')
        for x in tweet['tweets']:
            tags = tagger.TagText(x)
            row=''
            for w in tags:
                  a,b,c=w.split()
                  palabra=b+'/'+c+' '
                  row = row+palabra
            preproceso.append(row)

        #ajuste idioma español
        if language == 'es':
            tweet['preproceso']=preproceso
        #ajuste idioma inglés
        elif language == 'en':
            tweet['preprocess']=preproceso
        st.subheader(_('Preprocesamiento'))
        if language == 'es':
            st.write(tweet['preproceso'])
        elif language == 'en':
            st.write(tweet['preprocess'])

        if language == 'es':
            x= tweet.preproceso
        if language == 'en':
            x= tweet.preprocess
        y= tweet.label_num

        #división del conjunto de datos
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=test, train_size=train)

        #TF-IDF
        vect = TfidfVectorizer(token_pattern='\S+', ngram_range=(ngram_1, ngram_2),max_df=0.5, min_df=2, max_features = max_feat)
        x_train_dtm = vect.fit_transform(x_train)

        x_test_dtm = vect.transform(x_test)

        st.subheader(_('División del conjunto de datos'))
        st.markdown(_('Set de entrenamiento (80%):'))
        st.write(x_train_dtm.shape)
        st.markdown(_('Set de prueba (20%):'))
        st.write(x_test_dtm.shape)

        #creación bolsa de palabras
        st.subheader(_('Bolsa de palabras'))
        matriz= pd.DataFrame(x_train_dtm.toarray(), columns=vect.get_feature_names_out())
        st.write(matriz)

        #entrenamiento del clasificador
        nb_multi = MultinomialNB()
        nb_multir = nb_multi.fit(x_train_dtm, y_train)
        y_pred_class = nb_multi.predict(x_test_dtm)

        #cálculo de métricas
        st.subheader(_('Métricas'))
        st.markdown(_('Accuracy:'))
        st.info(accuracy_score(y_test, y_pred_class))
        st.markdown(_('Precision:'))
        st.info(precision_score(y_test, y_pred_class, average='weighted'))
        st.markdown(_('Recall:'))
        st.info(recall_score(y_test, y_pred_class, average='weighted'))
        st.markdown(_('F1-score:'))
        st.info(f1_score(y_test, y_pred_class, average='weighted'))

        #crear matriz de confusión
        matrix = confusion_matrix(y_test, y_pred_class)

        #crear marco de datos de pandas 
        dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
        
        return dataframe


def main():
    
    final=0

    #selección de idioma
    language = st.sidebar.selectbox('',['es', 'en'])

    #carga de ficheros de traducción
    try:
        localizator = gettext.translation('myapp', localedir='locales', languages=[language])
        localizator.install()
        _=localizator.gettext
        
    except:
        pass  

    #logo aplicación web
    col1, col2, col3 = st.columns(3)
    image = Image.open('traducir/logo.png')
    with col1:
        st.write(' ')
    with col2:
        st.image(image, width=250)

    st.title(_('Detección de odio y sesgos sociales'))
    st.markdown(_('Esta aplicación nos permite visualizar el proceso del sistema de minería de opiniones creado para la detección del sentimiento de odio y la identificicación de sesgos sociales como la discriminación racial y de género'))
    st.sidebar.header(_('Parámetros'))
    
    #opción vista previa dataset
    agree = st.sidebar.checkbox(_('Mostrar datos'))
    
    #método para poner los parametros en el sidebar
    def user_input_parameters():
        max_features = st.sidebar.slider('max_features', 1, 15, 3)
        test = st.sidebar.slider('test_size', 0.0, 1.0, 0.2)
        train = st.sidebar.slider('train_size', 0.0, 1.0, 0.8)
        if language == 'es':
            n_grams = st.sidebar.radio("Indique que tipo de n-grama desea utilizar:", ('unigramas', 'bigramas', 'ambos'))
        elif language == 'en':
            n_grams = st.sidebar.radio("Choose the type of n-gram:", ('unigram', 'bigram', 'both'))
        data = {'max_features': max_features,
                'test_size': test,
                'train_size': train,
                'ngrama': n_grams,
               }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el dataset deseado
    if language == 'es':
        option_dataset = ['odio', 'odio y discriminación racial', 'odio, discriminación racial y de género', 'importar CSV']
        dataset = st.sidebar.selectbox('¿Qué dataset desea usar?', option_dataset)
    elif language == 'en':
        option_dataset = ['hate', 'hate and racial discrimination', 'hate, racial and gender discrimination', 'import CSV']
        dataset = st.sidebar.selectbox('Which dataset do you want to use?', option_dataset)

    #importar CSV
    if dataset == 'importar CSV' or dataset == 'import CSV':
            st.subheader(_('Importar dataset'))
            if language == 'es':
                data_file = st.file_uploader('Importar CSV', type=['csv'])
            elif language == 'en':
                data_file = st.file_uploader('Import CSV', type=['csv'])
            if data_file is not None:
                        data_user = pd.read_csv(data_file, header=None, sep=';', names=['label', 'tweets'])
                        if agree:
                            if language == 'es':
                                st.subheader(f"Dataset: {data_file.name} ")
                                st.write('Dimensión del dataset: ' + str(data_user.shape[0]) + ' filas y ' + str(data_user.shape[1]) + ' columnas.')
                                st.dataframe(data_user, 2000)
                            elif language == 'en':
                                st.subheader(f"Dataset: {data_file.name} ")
                                st.write('Dataset dimension: ' + str(data_user.shape[0]) + ' rows and ' + str(data_user.shape[1]) + ' columns.')
                                st.dataframe(data_user, 2000)

    #escoger la herramienta de preprocesamiento
    option_herramienta = ['NLTK', 'TreeTagger']
    if language == 'es':
        herramienta = st.sidebar.selectbox('¿Qué herramienta desea usar?', option_herramienta)
    elif language == 'en':
        herramienta = st.sidebar.selectbox('What tool do you want to use?', option_herramienta)
    st.subheader(_('Parámetros del usuario'))
    if language == 'es':
        st.markdown(f"Herramienta seleccionada: {herramienta}")
    elif language == 'en':
        st.markdown(f"Selected tool: {herramienta}")
    st.write(df)

    #mostrar dataset seleccionado
    if agree:
            if dataset == 'odio' or dataset == 'hate':
                    seleccionado = pd.read_csv('traducir/total_tuits_odio.csv', header=None, sep=';', names=['label', 
'tweets'])
                    st.subheader(f"Dataset: {dataset} ")
                    if language == 'es':
                        st.write('Dimensión del dataset: ' + str(seleccionado.shape[0]) + ' filas y ' + str(seleccionado.shape[1]) + ' columnas.')
                    elif language == 'en':
                        st.write('Dataset dimension: ' + str(seleccionado.shape[0]) + ' rows and ' + str(seleccionado.shape[1]) + ' columns.')
                    st.dataframe(seleccionado, 2000)
                    st.markdown(filedownload(seleccionado, language), unsafe_allow_html=True)
            elif dataset == 'odio y discriminación racial' or dataset == 'hate and racial discrimination':
                    seleccionado = pd.read_csv('traducir/total_tuits_odio_1discriminacion.csv', header=None, sep=';', 
names=['label', 'tweets'])
                    st.subheader(f"Dataset: {dataset} ")
                    if language == 'es':
                        st.write('Dimensión del dataset: ' + str(seleccionado.shape[0]) + ' filas y ' + str(seleccionado.shape[1]) + ' columnas.')
                    elif language == 'en':
                        st.write('Dataset dimension: ' + str(seleccionado.shape[0]) + ' rows and ' + str(seleccionado.shape[1]) + ' columns.')
                    st.dataframe(seleccionado, 2000)
                    st.markdown(filedownload(seleccionado, language, final), unsafe_allow_html=True)
            elif dataset == 'odio, discriminación racial y de género' and dataset == 'hate, racial and gender discrimination':
                    seleccionado = pd.read_csv('traducir/total_tuits_odio_2discriminaciones.csv', header=None, sep=';', 
names=['label', 'tweets'])
                    st.subheader(f"Dataset: {dataset} ")
                    if language == 'es':
                        st.write('Dimensión del dataset: ' + str(seleccionado.shape[0]) + ' filas y ' + str(seleccionado.shape[1]) + ' columnas.')
                    elif language == 'en':
                        st.write('Dataset dimension: ' + str(seleccionado.shape[0]) + ' rows and ' + str(seleccionado.shape[1]) + ' columns.')
                    st.dataframe(seleccionado, 2000)
                    st.markdown(filedownload(seleccionado), unsafe_allow_html=True)

    #lanzar sistema de minería de opinión
    if st.button(_('Clasificar')):
        #mostrar etiquetado español
        st.subheader(_('Etiquetado sentimientos'))
        info = pd.DataFrame()
        if language == 'es':
            info['etiqueta'] = 0
            info.loc['no odio'] = 0
            solo_odio= ['no odio','odio']
            odio_racial=['no odio','odio','discriminación racial']
            odio_racial_genero=['no odio','odio','discriminación racial','discriminación de género']
            if dataset == 'odio':
                    dataset_clasificador = 'traducir/total_tuits_odio.csv'
                    class_names= solo_odio
                    info.loc['odio'] = 1
                    st.write(info)
            elif dataset == 'odio y discriminación racial':
                    dataset_clasificador = 'traducir/total_tuits_odio_1discriminacion.csv'
                    class_names=odio_racial
                    info.loc['odio'] = 1
                    info.loc['discriminación racial'] = 2
                    st.write(info)
            elif dataset == 'odio, discriminación racial y de género':
                    dataset_clasificador = 'traducir/total_tuits_odio_2discriminaciones.csv'
                    class_names=odio_racial_genero
                    info.loc['odio'] = 1
                    info.loc['discriminación racial'] = 2
                    info.loc['discriminación de género'] = 3
                    st.write(info)
            elif dataset == 'importar CSV':
                    dataset_clasificador = data_file.name
                    class_names = data_user.iloc[:, 0].unique()
                    detalles = pd.DataFrame(class_names)
                    detalles = detalles.rename(columns={0:'etiqueta'})
                    detalles['variable'] = detalles.index
                    detalles = detalles.set_index('etiqueta')
                    detalles = detalles.rename(columns={'variable':'etiqueta'})
                    detalles = detalles.reindex(index=['no odio','odio','discriminacion racial','discriminacion genero'])
                    detalles['etiqueta'] = np.arange(len(detalles))
                    st.write(detalles)
                    
        #mostrar etiquetado inglés
        elif language == 'en':
            info['label'] = 0
            info.loc['non-hate'] = 0
            solo_odio= ['non-hate','hate']
            odio_racial=['non-hate','hate','racial discrimination']
            odio_racial_genero=['non-hate','hate','racial discrimination','gender discrimination']
            if dataset == 'hate':
                    dataset_clasificador = 'traducir/total_tuits_odio.csv'
                    class_names= solo_odio
                    info.loc['hate'] = 1
                    st.write(info)
            elif dataset == 'hate and racial discrimination':
                    dataset_clasificador = 'traducir/total_tuits_odio_1discriminacion.csv'
                    class_names=odio_racial
                    info.loc['hate'] = 1
                    info.loc['racial discrimination'] = 2
                    st.write(info)
            elif dataset == 'hate, racial and gender discrimination':
                    dataset_clasificador = 'traducir/total_tuits_odio_2discriminaciones.csv'
                    class_names=odio_racial_genero
                    info.loc['hate'] = 1
                    info.loc['racial discrimination'] = 2
                    info.loc['gender discrimination'] = 3
                    st.write(info)
            elif dataset == 'import CSV':
                    dataset_clasificador = data_file.name
                    class_names = data_user.iloc[:, 0].unique()
                    detalles = pd.DataFrame(class_names)
                    detalles = detalles.rename(columns={0:'label'})
                    detalles['variable'] = detalles.index
                    detalles = detalles.set_index('label')
                    detalles = detalles.rename(columns={'variable':'label'})
                    detalles = detalles.reindex(index=['non-hate','odio','racial discrimination','gender discrimination'])
                    detalles['label'] = np.arange(len(detalles))
                    st.write(detalles)
             
       
        #convertir selección del usuario a parámetros
        if df.ngrama[0] == 'unigramas' or df.ngrama[0] == 'unigram':
                ngramas_1 = 1
                ngramas_2 = 1
        elif df.ngrama[0] == 'bigramas' or df.ngrama[0] == 'bigram':
                ngramas_1 = 2
                ngramas_2 = 2
        elif df.ngrama[0] == 'ambos' or df.ngrama[0] == 'both':
                ngramas_1 = 1
                ngramas_2 = 2

        if herramienta == 'NLTK':
                result = analisis_1(dataset_clasificador, df.max_features[0], ngramas_1, ngramas_2, class_names, df.test_size[0], df.train_size[0], language)
        elif herramienta == 'TreeTagger':
                result = analisis_2(dataset_clasificador, df.max_features[0], ngramas_1, ngramas_2, class_names, df.test_size[0], df.train_size[0], language)
                
        #calcular total de la muestra para representar la matriz de confusión
        if language == 'es':
            result['total_muestra'] = result.sum(axis=1)
        elif language == 'en':
            result['total_sample'] = result.sum(axis=1)
        st.subheader(_('Matriz de confusión'))
        st.bar_chart(result)
        final=1
        st.markdown(filedownload(result, language, final), unsafe_allow_html=True)
  
    

if __name__ == '__main__':
    main()


