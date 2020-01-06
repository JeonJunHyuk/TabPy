
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tabpy.tabpy_tools.client import Client

#데이터 읽기
train = pd.read_csv('c:/Users/JunHyuk/downloads/TabPy materials/LoanStatsFilter.csv')

#데이터 살짝 보기
train.info()
train.head()
pd.set_option('display.max_columns',50)
pd.set_option('display.width',500)
train = train[train.inactive_loans == 1]
train['purpose'].unique()
train.columns
#전처리
test = train.iloc[:,1:6]
test.columns
enc = preprocessing.LabelEncoder()
enc2 = preprocessing.LabelEncoder()
train['grade'] = enc.fit_transform(train['grade'])
train['purpose'] = enc2.fit_transform(train['purpose'])
train.columns
train.grade.unique()
train.purpose.unique()

# Separate the data into the class labels y and the feature variables X.
targets = train['bad_loans']
y = np.array(train['bad_loans']).astype(int) # 이 모델에서는 이렇게 형태를 만들어줘야하나봐
X = train.ix[:,1:6]  #http://yeyej.blogspot.com/2016/02/pandas-dataframe-iloc-loc-ix.html
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)  #stratify 찾아보자?

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train = scaler.fit_transform(X_train)   #train set에 fitting & transform
X_test = scaler.transform(X_test)         #fitting된 scaler로 transform

model = MLPClassifier(hidden_layer_sizes=(500,500,100), activation='relu', solver='adam', alpha=1e-5,
                                         batch_size='auto', learning_rate='constant', learning_rate_init=0.0001,
                                         power_t=0.5, max_iter=10, shuffle=True, random_state=None,
                                         tol=0.00001, verbose=True, warm_start=False, momentum=0.9,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                         beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:10])
probs = model.predict_proba(X_test)
print(probs[:10])


threshold = 0.2
threshold_preds = []
for i in range(len(probs)):
    if probs[i][1] >= threshold:
        threshold_preds.append(1)
    else:
        threshold_preds.append(0)

# mask = probs[:,1] > threshold
# thre_preds = probs[mask,1]
# probs[mask,1].tolist()
a = probs[:,1]
a[a>=threshold] = 1
a[a<threshold] = 0

np.where(probs[:,1]<threshold, 0, 1)

accuracy = metrics.accuracy_score(y_test, threshold_preds)
# accuracy = metrics.accuracy_score(y_test, thre_preds)
accuracy = metrics.accuracy_score(y_test, a)

print("The model produced {0}% accurate predictions.".format(accuracy*100))

print(metrics.classification_report(y_test, threshold_preds))
print(metrics.classification_report(y_test, a))

metrics.confusion_matrix(y_test, threshold_preds)
metrics.confusion_matrix(y_test, a)

def loanclassifierfull(_arg1, _arg2, _arg3, _arg4, _arg5):
    from pandas import DataFrame

    # Load data from tableau (brought in as lists) into a dictionary
    # Like I mentioned in my email, the columns get sorted alphabetically in this constructor
    # Adding the numbers sorts them correctly
    d = {'1-grade': _arg1, '2-income': _arg2, '3-sub_grade_num': _arg3, '4-purpose': _arg4, '5-dti': _arg5}
    # Convert the dictionary to a Pandas Dataframe
    df = DataFrame(data=d)

    # Transform categorical variables into numerical/continuous features
    df['1-grade'] = enc.transform(df['1-grade'])
    df['4-purpose'] = enc2.transform(df['4-purpose'])
    print(df.head())

    # This is the missing step from my first version
    # We need to scale the inputs to the Model or it will be totally off
    # Hope no one saw this
    # The scaler, since it's saved in the code, should be [pickled automatically by TabPy and available for reuse
    # This should also be the case for the feature encoder above
    df = scaler.transform(df)

    # Use the loaded model to develop predictions for the new data from Tableau
    probs = model.predict_proba(df)
    return probs[:,1].tolist()

func_probs =loanclassifierfull(test.iloc[:,0],test.iloc[:,1],test.iloc[:,2],test.iloc[:,3],test.iloc[:,4])
print('Calc Results Come After This')
print(func_probs[:10])

client = Client('http://localhost:9004/')

client.deploy('loanclassifierfull', loanclassifierfull,
              'Returns the probablility that a loan will result in a bad loan based on its Grade, Income, '
              'SubGradeNum, Purpose, and DTI', override=True)