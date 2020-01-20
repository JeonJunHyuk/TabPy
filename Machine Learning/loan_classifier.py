import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

# 데이터 읽기
train = pd.read_csv('c:/Users/JunHyuk/downloads/TabPy materials/LoanStatsFilter.csv')

# 데이터 살짝 보기
train.info()
train.head()

train['grade'].unique()
train['purpose'].unique()
train['bad_loans'].unique()
train['inactive_loans'].unique()
# bad_loans는 target
# grade, purpose는 label encoder
# id, inactive_loans 는 버리기


train.drop(['id', 'inactive_loans'], axis=1, inplace=True)
# 또는 train = train.iloc[:,:8]

enc = preprocessing.LabelEncoder()
enc2 = preprocessing.LabelEncoder()
train['grade'] = enc.fit_transform(train['grade'])
train['purpose'] = enc2.fit_transform(train['purpose'])

train.grade.unique()
train.purpose.unique()

onehot1 = preprocessing.OneHotEncoder()
onehot2 = preprocessing.OneHotEncoder()

oh_grade = onehot1.fit_transform(np.array(train['grade']).reshape(-1, 1)).toarray()
oh_purpose = onehot2.fit_transform(np.array(train['purpose']).reshape(-1,1)).toarray()

oh_grade_df = pd.DataFrame(oh_grade, columns=['A','B','C','D','E','F','G'])
oh_purpose_df = pd.DataFrame(oh_purpose)

oh_grade_df.head()
oh_purpose_df.head()

train = pd.concat([train,oh_grade_df,oh_purpose_df], axis=1)
train

train.drop(['grade','purpose'],axis=1,inplace=True)
train

train.drop(['A',0], axis=1, inplace=True)
train

# target을 지정하고 나머지는 feature
y = np.array(train['bad_loans'])  # index가 없게
X = train.drop('bad_loans', axis=1)
# .loc: label based indexing, .iloc: positional indexing

# train, test set 8:2로 나누기.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=None, stratify=y)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_train = scaler.fit_transform(X_train)  # train set에 fitting & transform
X_test = scaler.transform(X_test)  # fitting 된 scaler 로 transform

model = MLPClassifier(hidden_layer_sizes=(500, 500, 100),
                      activation='relu', solver='adam', alpha=1e-5,
                      batch_size='auto', learning_rate='constant',
                      learning_rate_init=0.0001,
                      power_t=0.5, max_iter=10, shuffle=True, random_state=None,
                      tol=0.00001, verbose=True, warm_start=False, momentum=0.9,
                      nesterovs_momentum=True, early_stopping=False,
                      validation_fraction=0.1,
                      beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions[:10]
probs = model.predict_proba(X_test)
probs[:10]

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
a = probs[:, 1]
a[a >= threshold] = 1
a[a < threshold] = 0
a

b = np.where(probs[:, 1] < threshold, 0, 1)

accuracy = metrics.accuracy_score(y_test, predictions)
accuracy

accuracy = metrics.accuracy_score(y_test, threshold_preds)
accuracy
accuracy = metrics.accuracy_score(y_test, a)
accuracy
accuracy = metrics.accuracy_score(y_test, b)
accuracy





print("The model has {0}% accuracy.".format(accuracy * 100))

print(metrics.classification_report(y_test, a))

metrics.confusion_matrix(y_test, a)


def loanclassifierfull(_arg1, _arg2, _arg3, _arg4, _arg5):
    from pandas import DataFrame

    # 태블로에서 list로 받으면서 dictionary로 저장
    # 알파벳 순으로 정렬되니 순서대로 되도록 column 명에 숫자 붙임.
    d = {'1-grade': _arg1, '2-income': _arg2,
         '3-sub_grade_num': _arg3, '4-purpose': _arg4, '5-dti': _arg5}

    df = DataFrame(data=d)  # dataframe 에 저장

    # train set 에서 fit 했던 enc, enc2, scaler.
    df['1-grade'] = enc.transform(df['1-grade'])
    df['4-purpose'] = enc2.transform(df['4-purpose'])

    new_grade = onehot1.transform(np.array(df['1-grade']).reshape(-1, 1)).toarray()
    new_purpose = onehot2.transform(np.array(df['4-purpose']).reshape(-1, 1)).toarray()

    new_grade_df = pd.DataFrame(new_grade, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    new_purpose_df = pd.DataFrame(new_purpose)
    df = pd.concat([df, new_grade_df, new_purpose_df], axis=1)
    df.drop(['1-grade', '4-purpose'], axis=1, inplace=True)
    df.drop(['A', 0], axis=1, inplace=True)

    df = scaler.transform(df)

    # train set으로 만든 모델로 예측
    probs = model.predict_proba(df)
    return probs[:, 1].tolist()


test = pd.read_csv('c:/Users/JunHyuk/downloads/TabPy materials/LoanStatsFilter.csv')
test = test.iloc[:, 1:6]
test.head()
func_probs = loanclassifierfull(test.iloc[:, 0], test.iloc[:, 1],
                                test.iloc[:, 2], test.iloc[:, 3], test.iloc[:, 4])

print('Calc Results Come After This')
print(func_probs[:10])

from tabpy.tabpy_tools.client import Client

client = Client('http://localhost:9004/')
client.deploy('loanclassifierfull', loanclassifierfull,
              'Returns the probablility that a loan will result in '
              'a bad loan based on its Grade, Income, '
              'SubGradeNum, Purpose, and DTI', override=True)
