from sklearn.metrics import roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

label = pd.read_table('crm_order_train_data20180104.txt', header=None)
label = label.iloc[:, :2]
label.columns = ['uid', 'label']

positive = label[label['label'] == 1].reset_index()
negative = label[label['label'] == 0].reset_index()

index_arrange = list(range(negative.shape[0]))
np.random.shuffle(index_arrange)
negative_new = negative.iloc[index_arrange[:positive.shape[0]], :].reset_index()
train = pd.concat([negative_new, positive], ignore_index=True, axis=0)
train = train[['uid', 'label']]

# user_train = []
# item_train = []
# f = open('pred.state')
# cnt = 0
# for i in f.readlines():
#     if i.strip().split('\t')[0] not in user_train:
#         user_train.append(i.strip().split('\t')[0])
#         item_train.append(i.strip().split('\t')[1:])
#     print('we have try %s times' % cnt)
#     cnt = cnt + 1

user_item = {}
f = open('pred.state')
cnt = 0
for i in f.readlines():
    if i.strip().split('\t')[0] not in user_item.keys():
        user_item[i.strip().split('\t')[0]] = i.strip().split('\t')[1:]
    else:
        user_item[i.strip().split('\t')[0]].append(i.strip().split('\t')[1:])
    print('we have try %s times' % cnt)
    cnt = cnt + 1

# user_test = []
# item_test = []
# f = open('predtest.state')
# cnt = 0
# for i in f.readlines():
#     if i.strip().split('\t')[0] not in user_test:
#         user_test.append(i.strip().split('\t')[0])
#         item_test.append(i.strip().split('\t')[1:])
#     print('we have try %s times' % cnt)
#     cnt = cnt + 1

user_item_test = {}
f = open('predtest.state')
cnt = 0
for i in f.readlines():
    if i.strip().split('\t')[0] not in user_item_test.keys():
        user_item_test[i.strip().split('\t')[0]] = i.strip().split('\t')[1:]
    else:
        user_item_test[i.strip().split('\t')[0]].append(i.strip().split('\t')[1:])
    print('we have try %s times' % cnt)
    cnt = cnt + 1

uid = np.array(train['uid'])


uid_data = []
item_data = []
for i in uid:
    if str(i) in user_item.keys():
        uid_data.append(i)
        item_data.append(user_item[str(i)])
        print('we have found %s key' % i)

uid_data = pd.DataFrame(uid_data)
uid_data.columns = ['uid']

item_data1 = pd.DataFrame(item_data)
item_data2 = item_data1.iloc[:, 100:]
m, n = item_data2.shape
value_bind = pd.DataFrame()
for i in range(n):
    for j in range(m):
        value = pd.DataFrame(item_data2.iloc[j, i])
        value_bind = pd.concat([value_bind, value], axis=1)
        print('we have got the %s col %s row' % (i, j))

value0 = value_bind.iloc[:, :4968].T
value1 = value_bind.iloc[:, 4968:4968 * 2].T
value2 = value_bind.iloc[:, 4968 * 2:4968 * 3].T
value3 = value_bind.iloc[:, 4968 * 3:].T
value = item_data1.iloc[:, :100]

value0 = value0.reset_index(drop=True)
value1 = value1.reset_index(drop=True)
value2 = value2.reset_index(drop=True)
value3 = value3.reset_index(drop=True)

item_all = pd.concat([value, value0, value1, value2, value3], axis=1)
colnames = ['f' + str(x) for x in range(500)]
item_all.columns = colnames

features = item_all
targets_values = uid_data.merge(train,on='uid',how = 'left')['label']

train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=0)

lr = LogisticRegression(C=1, penalty='l1', solver='liblinear', multi_class='ovr')
model_lr = lr.fit(train_X, train_y)
y_test_lr = model_lr.predict_proba(test_X)[:, 1]
roc_auc_score(test_y, y_test_lr)



###########################################################################################################################################################################
from sklearn.metrics import roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

label = pd.read_table('crm_order_train_data20180104.txt', header=None)
label = label.iloc[:, :2]
label.columns = ['uid', 'label']

positive = label[label['label'] == 1].reset_index()
negative = label[label['label'] == 0].reset_index()

index_arrange = list(range(negative.shape[0]))
np.random.shuffle(index_arrange)
negative_new = negative.iloc[index_arrange[:positive.shape[0]], :].reset_index()
train = pd.concat([negative_new, positive], ignore_index=True, axis=0)
train = train[['uid', 'label']]

user_item = {}
f = open('predtrain.state')
cnt = 0
for i in f.readlines():
    if i.strip().split('\t')[0] not in user_item.keys():
        user_item[i.strip().split('\t')[0]] = i.strip().split('\t')[1:]
    else:
        user_item[i.strip().split('\t')[0]].append(i.strip().split('\t')[1:])
    print('we have try %s times' % cnt)
    cnt = cnt + 1

user_item_test = {}
f = open('predtest.state')
cnt = 0
for i in f.readlines():
    if i.strip().split('\t')[0] not in user_item_test.keys():
        user_item_test[i.strip().split('\t')[0]] = i.strip().split('\t')[1:]
    else:
        user_item_test[i.strip().split('\t')[0]].append(i.strip().split('\t')[1:])
    print('we have try %s times' % cnt)
    cnt = cnt + 1

uid = np.array(train['uid'])


uid_data = []
item_data = []
for i in uid:
    if str(i) in user_item.keys():
        uid_data.append(i)
        item_data.append(user_item[str(i)])
        print('we have found %s key' % i)
