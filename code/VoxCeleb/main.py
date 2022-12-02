import pickle
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn import metrics
import datetime
from operator import itemgetter

import time
import warnings
warnings.filterwarnings("ignore")
random.seed(0)

def create_dataset(path):
    # load male dataset 
    with open(path+"male_data.pkl", 'rb') as f:
        male_data = pickle.load(f)
    with open(path+"female_data.pkl", 'rb') as f:
        female_data = pickle.load(f)
    data = [{"x": d, "y": 0} for d in male_data]
    data += [{"x": d, "y": 1} for d in female_data]
    random.shuffle(data)

    new_data_x = []
    new_data_y = []
    for d in data:
        new_data_x.append(d["x"])
        new_data_y.append(d["y"])
    
    return np.array(new_data_x), np.array(new_data_y)


def get_optimized_model(train_x, train_y, test_x, test_y):

    start = time.time()

    # decision tree
    try:
        #decision tree
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(train_x,train_y)
        scores = cross_val_score(classifier2, test_x, test_y,cv=5)
        print('Decision tree accuracy (+/-) %s'%(str(scores.std())))
        c2=scores.mean()
        c2s=scores.std()
        print(c2)
    except:
        c2=0
        c2s=0

    try:
        classifier3 = GaussianNB()
        classifier3.fit(train_x, train_y)
        scores = cross_val_score(classifier3, test_x, test_y,cv=5)
        print('Gaussian NB accuracy (+/-) %s'%(str(scores.std())))
        c3=scores.mean()
        c3s=scores.std()
        print(c3)
    except:
        c3=0
        c3s=0

    try:
        #svc 
        classifier4 = SVC()
        classifier4.fit(train_x,train_y)
        scores=cross_val_score(classifier4, test_x, test_y,cv=5)
        print('SKlearn classifier accuracy (+/-) %s'%(str(scores.std())))
        c4=scores.mean()
        c4s=scores.std()
        print(c4)
    except:
        c4=0
        c4s=0

    try:
        #adaboost
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_x, train_y)
        scores = cross_val_score(classifier6, test_x, test_y,cv=5)
        print('Adaboost classifier accuracy (+/-) %s'%(str(scores.std())))
        c6=scores.mean()
        c6s=scores.std()
        print(c6)
    except:
        c6=0
        c6s=0

    try:
        #gradient boosting 
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_x, train_y)
        scores = cross_val_score(classifier7, test_x, test_y,cv=5)
        print('Gradient boosting accuracy (+/-) %s'%(str(scores.std())))
        c7=scores.mean()
        c7s=scores.std()
        print(c7)
    except:
        c7=0
        c7s=0

    try:
        #logistic regression
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(train_x, train_y)
        scores = cross_val_score(classifier8, test_x, test_y,cv=5)
        print('Logistic regression accuracy (+/-) %s'%(str(scores.std())))
        c8=scores.mean()
        c8s=scores.std()
        print(c8)
    except:
        c8=0
        c8s=0

    try:
        #voting 
        classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(train_x, train_y)
        scores = cross_val_score(classifier9, test_x, test_y,cv=5)
        print('Hard voting accuracy (+/-) %s'%(str(scores.std())))
        c9=scores.mean()
        c9s=scores.std()
        print(c9)
    except:
        c9=0
        c9s=0

    try:
        #knn
        classifier10=KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(train_x, train_y)
        scores = cross_val_score(classifier10, test_x, test_y,cv=5)
        print('K Nearest Neighbors accuracy (+/-) %s'%(str(scores.std())))
        c10=scores.mean()
        c10s=scores.std()
        print(c10)
    except:
        c10=0
        c10s=0

    try:
        #randomforest
        classifier11=RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=2, random_state=0)
        classifier11.fit(train_x, train_y)
        scores = cross_val_score(classifier11, test_x, test_y,cv=5)
        print('Random forest accuracy (+/-) %s'%(str(scores.std())))
        c11=scores.mean()
        c11s=scores.std()
        print(c11)
    except:
        c11=0
        c11s=0

    try:
    ##        #svm
        classifier12 = svm.SVC(kernel='rbf', C = 1.0)
        classifier12.fit(train_x, train_y)
        scores = cross_val_score(classifier12, test_x, test_y,cv=5)
        print('svm accuracy (+/-) %s'%(str(scores.std())))
        c12=scores.mean()
        c12s=scores.std()
        print(c12)
    except:
        c12=0
        c12s=0

    maxacc=max([c2,c3,c4,c6,c7,c8,c9,c10,c11,c12])

    if maxacc==c2:
        # print('most accurate classifier is Decision Tree'+'with %s'%(selectedfeature))
        classifiername='decision-tree'
        classifier=classifier2
    elif maxacc==c3:
        # print('most accurate classifier is Gaussian NB'+'with %s'%(selectedfeature))
        classifiername='gaussian-nb'
        classifier=classifier3
    elif maxacc==c4:
        # print('most accurate classifier is SK Learn'+'with %s'%(selectedfeature))
        classifiername='sk'
        classifier=classifier4
    #can stop here (c6-c10)
    elif maxacc==c6:
        # print('most accuracate classifier is Adaboost classifier'+'with %s'%(selectedfeature))
        classifiername='adaboost'
        classifier=classifier6
    elif maxacc==c7:
        # print('most accurate classifier is Gradient Boosting '+'with %s'%(selectedfeature))
        classifiername='graidentboost'
        classifier=classifier7
    elif maxacc==c8:
        # print('most accurate classifier is Logistic Regression '+'with %s'%(selectedfeature))
        classifiername='logistic_regression'
        classifier=classifier8
    elif maxacc==c9:
        # print('most accurate classifier is Hard Voting '+'with %s'%(selectedfeature))
        classifiername='hardvoting'
        classifier=classifier9
    elif maxacc==c10:
        # print('most accurate classifier is K nearest neighbors '+'with %s'%(selectedfeature))
        classifiername='knn'
        classifier=classifier10
    elif maxacc==c11:
        # print('most accurate classifier is Random forest '+'with %s'%(selectedfeature))
        classifiername='randomforest'
        classifier=classifier11
    elif maxacc==c12:
        # print('most accurate classifier is SVM '+' with %s'%(selectedfeature))
        classifiername='svm'
        classifier=classifier12

    modeltypes=['decision-tree','gaussian-nb','sk','adaboost','gradient boosting','logistic regression','hard voting','knn','random forest','svm']
    accuracym=[c2,c3,c4,c6,c7,c8,c9,c10,c11,c12]
    accuracys=[c2s,c3s,c4s,c6s,c7s,c8s,c9s,c10s,c11s,c12s]
    model_accuracy=list()
    for i in range(len(modeltypes)):
        model_accuracy.append([modeltypes[i],accuracym[i],accuracys[i]])

    model_accuracy.sort(key=itemgetter(1))
    endlen=len(model_accuracy)

    print('saving classifier to disk')
    f=open("model"+'.pickle','wb')
    pickle.dump(classifier,f)
    f.close()

    end=time.time()

    execution=end-start
    
    print('summarizing session...')

    accstring=''
    
    for i in range(len(model_accuracy)):
        accstring=accstring+'%s: %s (+/- %s)\n'%(str(model_accuracy[i][0]),str(model_accuracy[i][1]),str(model_accuracy[i][2]))

    training=len(train_x)
    testing=len(test_x)
    
    
    data={
        'model': "model",
        'modeltype':model_accuracy[len(model_accuracy)-1][0],
        'accuracy':model_accuracy[len(model_accuracy)-1][1],
        'deviation':model_accuracy[len(model_accuracy)-1][2]
        }
    
    return [classifier, model_accuracy[endlen-1],  data]

if __name__=="__main__":
    X, Y = create_dataset("../../data/VoxCeleb/")
    print("dataset created")
    X_trn, X_tst, Y_trn,  Y_tst = train_test_split(X,Y, random_state=42, test_size = 0.2)
    res = get_optimized_model(X_trn, Y_trn, X_tst, Y_tst)
