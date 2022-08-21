import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
#from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)



model1 = pickle.load(open('omicoron_naive.pkl','rb'))
model2= pickle.load(open('omicoron_random_forest.pkl','rb'))
model3= pickle.load(open('omicoron_knn.pkl','rb'))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_omicron_dataset.csv')
corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()


@app.route('/')
def check():
    return render_template("check.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')

@app.route('/new')
def new():
    return render_template("new.html")

  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    text = request.args.get('text')
    
    text=[text]

    input_data = cv.transform(text).toarray()

    #prediction = model.predict(input_data)

    #input_pred = input_pred.astype(int)
    Model = (request.args.get('Model'))

    if Model=="Naive Bayes Classifier":
      prediction = model1.predict(input_data)

  
      
    elif Model=="KNN Classifer":
      prediction = model2.predict(input_data)

    elif Model=="RANdom Forest Classifer":
      prediction = model3.predict(input_data)

    



    
    if prediction[0]==2:
      return render_template('index.html', prediction_text='Text is Positive')
      
    elif prediction[0]==1:    
      return render_template('index.html', prediction_text='Text is Negative')
    else:
      return render_template('index.html', prediction_text='Text is Netural')

app.run()