import pandas as pd
import numpy
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
#Lectura del dataset
dfreview = pd.read_csv('IMDB Dataset.csv')

#slices o particiones para crear un set desbalanceado
dfpositivos = dfreview[dfreview['sentiment']=='positive'][:9000]
dfnegativos = dfreview[dfreview['sentiment']=='negative'][:1000]
dfreviewdes = pd.concat([dfpositivos,dfnegativos])

#Se hace un undersampling para balancear el dataset partido
rus = RandomUnderSampler()
dfreviewbal, dfreviewbal['sentiment'] = rus.fit_resample(dfreviewdes[['review']],dfreviewdes['sentiment'])

#se divide en un conjunto de entrenamiento y otro de prueba
train, test=train_test_split(dfreviewbal, test_size=0.33, random_state=42)

#Se llenan los conjuntos de entrenamiento y prueba en valor(review) y output(sentiment)
trainX, trainY=train['review'], train['sentiment']
testX, testY=test['review'], test['sentiment']

#ejemplo de manejo de texto a representacion numerica
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]
#Ejemplo para crear una matriz de conteo de frecuencias
df = pd.DataFrame({'review': ['review1', 'review2'], 'text':text})
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names_out())

#ejemplo para crear una matriz con valores tfidf
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
df_dtm = pd.DataFrame(tfidf_matrix.toarray(), index=df['review'].values, columns=tfidf.get_feature_names_out())

#creamos y llenamos los conjuntos de valores que vamos a usar para los modelos y creamos nuestra bolsa de palabras en base al conjunto de entrenamiento
tfidf = TfidfVectorizer(stop_words='english')
trainXVector = tfidf.fit_transform(trainX)
testXVector = tfidf.transform(testX)

#SVM
svc = SVC(kernel='linear')
svc.fit(trainXVector,trainY)

#save model
pickle.dump(svc, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('wordvector.mkl','wb'))