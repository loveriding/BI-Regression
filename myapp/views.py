from django.shortcuts import render
import numpy as np 
import pandas as pd 
from sklearn import tree
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import graphviz

def index(request):
    if "GET" == request.method:
        return render(request, 'myapp/index.html', {})
    csv_file = request.FILES["csv_file"]
    if not csv_file.name.endswith('.csv'):
        return render(request, 'myapp/index.html',{})
    
    #processing data if file input is csv file
    data2 = pd.read_csv("hour.csv")
    Y2 = data2['cnt']
    X = data2.drop(['instant','dteday','cnt'], axis=1)
    # chọn những colum cần dùng
    X3 = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered']
    # split the dataset
    X2 = X.values[:, 0:13]  
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.3, random_state = 10) 
    ## Performing training 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 3, min_samples_leaf = 5) 
    # make predictions 
    clf_entropy.fit(X2_train, y2_train)
    y_pred = clf_entropy.predict(X2_test)
    dot_data = tree.export_graphviz(clf_entropy, out_file = None,feature_names=X3 )
    graph = graphviz.Source(dot_data, format="svg")
    return render(request, 'myapp/index.html',{"csv_data": graph.view})








