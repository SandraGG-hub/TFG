#script para extraccion de tuits 
import tweepy
import csv
import pandas as pd
import unicodedata
import numpy as np
import preprocessor as p

#credenciales
consumer_key = 'GjCs2WACwgfXYxVEg52fVWqcy'
consumer_secret = '1JGsFNBJ6r5GQE1xrAMPCgIHXupXIgE8RStpG34i2ymCvo67c8'
access_token = '1491084910697844744-AonGTPepyFMotpEkCAOqTdqyTSVdgP'
access_token_secret = 'JBaLUPRYnWXhPCLnrdVXzPxw8CuAXNRlSLxcNwrs8lD42'

#autenticacion
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#lista palabras clave sentimiento de odio
lista_clave=['gentuza','puto facha','puta facha','puta fascista','puto fascista',
             'hijo de puta','hija de puta','hijadeputa','hijoputa','hijaputa','hijasdeputa','hijodeputa',
             'hijosdeputa','asco de gente','ascazo de gente','puta feminazi','maldita feminazi','das asco',
             'imbécil','gilipollas','malnacido','malnacida','bastardo','bastarada',
             'eres asqueroso','eres asquerosa','idiota','maricón','niñato','niñata','eres muy tont',
             'tu eres tont', 'sois basura','eres basura','miserable',
             'impresentable','vomitiv','calaña','indeseable','alimaña','estupido',
             'cantamañanas','tipejo','tipeja','gente repugnante','persona repugnante','mamarrach',
             'sinverguenza','subnormal','eres un mierda','eres una mierda', 'bocachancla',
             'lameculos','parguela','cenutrio','tolai','malfollao','malfollado','mal follado',
             'malfollada','mal follada','de los cojones','panoli','eres retrasado','eres retrasada',
             'sabandija','lerd','retrasada mental','mongol','cretin','ojala sufras',
             'ojala mueras', 'muerete','puto comunista','puto nazi','puta nazi',
             'puto baboso','inepto','eres mentiros','eres cobarde','eres fals',
             'eres impresentable','vete a tomar por culo','comerrabo','marimacho',
             'maricon','bollera','ignorante','de mierda','hdp','put','maldito',
             'asqueroso','callate','marica','mariposon','feminazi',
             'corrupto','corruptos','ladron','ladrones','podemita',
             'mariquita','soplanucas','soplapollas','cabezahueca',
             'arpía','babos','depravad','escoria','huevón','jodid',
             'mamón','odios','comeverga','chupapija','facha','fachas',
             'delincuentes','franquista','franquistas','muerdealmohada','travelo',
             'mermao\ndesviac', 'mermao\ndesviad', 'mermao\npervert',
             'mermao\npervers', 'mermao\ndepravac','mermao\ndepravad', 'mermao\npromiscu', 'mermao\nlibertin', 'mermao\nenferm',
             'mermao\nput', 'mermao\nmaldit', 'mermao\nsuci', 'mermao\npluma', 'mermao\ntijer', 'mermao\ncoño', 'mermao\npotorro',
             'mermao\npierdeaceit', 'mermao\nmierda', 'mermao\nbasura', 'mermao\ngentuza', 'mermao\nasco', 'mermao\nlacra',
             'mermao\nescoria', 'mermao\ncontagi', 'mermao\ndestroz', 'mermao\nreventar', 'mermao\nrevient', 'mermao\nmata',
             'mermao\nextermin','payas\ndesviac', 'payas\ndesviad', 'payas\npervert',
             'payas\npervers', 'payas\ndepravac','payas\ndepravad', 'payas\npromiscu', 'payas\nlibertin', 'payas\nenferm',
             'payas\nput', 'payas\nmaldit', 'payas\nsuci', 'payas\npluma', 'payas\ntijer', 'payas\ncoño', 'payas\npotorro',
             'payas\npierdeaceit', 'payas\nmierda', 'payas\nbasura', 'payas\ngentuza', 'payas\nasco', 'payas\nlacra',
             'payas\nescoria', 'payas\ncontagi', 'payas\ndestroz', 'payas\nreventar', 'payas\nrevient', 'payas\nmata',
             'payas\nextermin','cabron\ndesviac', 'cabron\ndesviad', 'cabron\npervert',
             'cabron\npervers', 'cabron\ndepravac','cabron\ndepravad', 'cabron\npromiscu', 'cabron\nlibertin', 'cabron\nenferm',
             'cabron\nput', 'cabron\nmaldit', 'cabron\nsuci', 'cabron\npluma', 'cabron\ntijer', 'cabron\ncoño', 'cabron\npotorro',
             'cabron\npierdeaceit', 'cabron\nmierda', 'cabron\nbasura', 'cabron\ngentuza', 'cabron\nasco', 'cabron\nlacra',
             'cabron\nescoria', 'cabron\ncontagi', 'cabron\ndestroz', 'cabron\nreventar', 'cabron\nrevient', 'cabron\nmata',
             'cabron\nextermin','gay\nvicio','gay\ndesviad','gay\npervert','gay\npervers','gay\ndepravad','gay\promiscu',
             'gay\nenferm','gay\nput','gay\nmaldit','gay\nsuci','gay\npluma','gay\ncoño','gay\nmierda','gay\nbasura','gay\ngentuza','gay\nasco',
             'gay\nescoria','comerabo', 'comepolla', 'chupapolla', 'machirul', 'marimach', 'hermafrodit','gay\ndesviac',
             'gay\ndepravac', 'gay\npromiscu', 'gay\nlibertin','gay\ntijer', 'gay\npotorro','gay\npierdeaceit', 'gay\nlacra',
             'gay\ncontagi', 'gay\ndestroz', 'gay\nreventar', 'gay\nrevient', 'gay\nmata',
             'gay\nextermin', 'maric\ndesviac', 'maric\ndesviad', 'maric\npervert', 'maric\npervers',
             'maric\ndepravac', 'maric\ndepravad', 'maric\npromiscu', 'maric\nlibertin', 'maric\nenferm',
             'maric\nput', 'maric\nmaldit', 'maric\nsuci', 'maric\npluma', 'maric\ntijer', 'maric\ncoño',
             'maric\npotorro', 'maric\npierdeaceit', 'maric\nmierda', 'maric\nbasura', 'maric\n-gentuza',
             'maric\nasco', 'maric\nlacra', 'maric\nescoria', 'maric\ncontagi', 'maric\ndestroz','maric\nreventar',
             'maric\nrevient', 'maric\nmata', 'maric\nextermin', 'mariq\ndesviac','mariq\ndesviad',
             'mariq\npervert', 'mariq\npervers', 'mariq\ndepravac', 'mariq\ndepravad','mariq\npromiscu', 'mariq\nlibertin',
             'mariq\nenferm', 'mariq\nput', 'mariq\nmaldit', 'mariq\nsuci', 'mariq\npluma', 'mariq\ntijer',
             'mariq\ncoño', 'mariq\npotorro', 'mariq\npierdeaceit', 'mariq\nmierda', 'mariq\nbasura', 'mariq\ngentuza',
             'mariq\nasco', 'mariq\nlacra','mariq\nescoria', 'mariq\ncontagi', 'mariq\ndestroz', 'mariq\nreventar',
             'mariq\nrevient','mariq\nmata', 'mariq\nextermin', 'lesbi\ndesviac', 'lesbi\ndesviad', 'lesbi\npervert',
             'lesbi\npervers', 'lesbi\ndepravac', 'lesbi\ndepravad', 'lesbi\npromiscu', 'lesbi\nlibertin','lesbi\nput',
             'lesbi\nmaldit', 'lesbi\nsuci', 'lesbi\npluma', 'lesbi\ntijer', 'lesbi\ncoño', 'lesbi\npotorro', 'lesbi\npierdeaceit',
             'lesbi\nmierda', 'lesbi\nbasura', 'lesbi\ngentuza', 'lesbi\nasco', 'lesbi\nlacra', 'lesbi\nescoria', 'lesbi\ncontagi',
             'lesbi\ndestroz', 'lesbi\nreventar', 'lesbi\nrevient', 'lesbi\nmata', 'lesbi\nextermin', 'trans\ndesviac',
             'trans\ndesviad','trans\npervert', 'trans\npervers', 'trans\ndepravac', 'trans\ndepravad', 'trans\npromiscu',
             'trans\nlibertin', 'trans\nenferm', 'trans\nput', 'trans\nmaldit', 'trans\nsuci', 'trans\npluma', 'trans\ntijer',
             'trans\ncoño', 'trans\npotorro', 'trans\npierdeaceit', 'trans\nmierda', 'trans\nbasura', 'trans\ngentuza',
             'trans\nasco', 'trans\nlacra', 'trans\nescoria', 'trans\ncontagi', 'trans\ndestroz', 'trans\nreventar',
             'trans\nrevient', 'trans\nmata', 'trans\nextermin', 'drag\ndesviac', 'drag\ndesviad', 'drag\npervert',
             'drag\npervers', 'drag\ndepravac','drag\ndepravad', 'drag\npromiscu', 'drag\nlibertin', 'drag\nenferm',
             'drag\nput', 'drag\nmaldit', 'drag\nsuci', 'drag\npluma', 'drag\ntijer', 'drag\ncoño', 'drag\npotorro',
             'drag\npierdeaceit', 'drag\nmierda', 'drag\nbasura', 'drag\ngentuza', 'drag\nasco', 'drag\nlacra',
             'drag\nescoria', 'drag\ncontagi', 'drag\ndestroz', 'drag\nreventar', 'drag\nrevient', 'drag\nmata',
             'drag\nextermin']
         

#lista clave para discriminación racial
#lista_clave_discriminacion_racial=['puto negro','puta negra',
            #'negrata','esclavo','inmigrante ladron','sudaca','negro de mierda',
            #'escoria humana','sin papeles','guiri de mierda','puto moro',
            #'rumano de mierda','moro de mierda','moro cabron','panchi','basura humana']

#lista clave para discrimianción de género
#lista_clave_discriminacion_genero=['puta feminista','maldita feminista','tipeja',
             #'malfollada','mal follada','ramera','buscona','calientapollas','putita',
             #'perra','zorra','puta','guarr','fulana','sargenta','bruja','verdulera','golfa',
             #'eso te pasa por ser mujer']


#controlar insultos con sarcasmo
def filtrado(cadena):
    cadena_min=cadena.lower()
    if (cadena_min.find('jaja')==-1) and (cadena_min.find('jeje')==-1) and (cadena_min.find('soy')==-1) and (cadena_min.find('estoy')==-1):
        return True

#eliminar tildes
def limpiar(cadena):
    s = ''.join((c for c in unicodedata.normalize('NFD',cadena) if unicodedata.category(c) != 'Mn'))
    return s

#busqueda de tweets y extraccion a csv
with open('extraertuits_odio.csv', 'w', newline='', encoding='utf-8') as csvFile:
     csvWriter = csv.writer(csvFile)
     for i in range(0,len(lista_clave)):
         for tweet in tweepy.Cursor(api.search_tweets,q=lista_clave[i],lang="es").items(100):
                if (filtrado(tweet.text)) and (not tweet.retweeted) and ('RT @' not in tweet.text) and ('https://' not in tweet.text):
                        print (tweet.created_at, limpiar(p.clean(tweet.text.replace('#', ''))))
                        csvWriter.writerow(['odio', limpiar(p.clean(tweet.text.replace('#', '')))])

