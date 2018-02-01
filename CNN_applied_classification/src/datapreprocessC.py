import pandas as pd
import numpy as np
import pickle

# reload data
def reload_feature(data, col1, col2):
    # value = data.groupby('SessionId')['ItemId']
    value = data.groupby(col1)[col2]
    result = []
    for i, j in value:
        result.append([[i], list(j)])
        print('we have get %s' % i)
    result = np.array(result)
    return result


def reload_feature_length(data, brandmap=None, count=20):
    result = {}
    for i in range(len(data)):
        result[data.take(i, 0)[0][0]] = data.take(i, 0)[1]
        print('we have got the %s times' % i)
    for j in result.keys():
        if len(result[j]) < count:
            for cnt in range(count - len(result[j])):
                result[j].insert(len(result[j]) + cnt, brandmap['<Nope>'])
        else:
            result[j] = result[j][-count:]
        print('we have got the %s columns fixed' % j)
    return result


def combind_data(data1, data2, data3):
    data = []
    for key in data1.keys():
        data.append(np.array([key, data1[key], data2[key], data3[key]], dtype=object))
        print('we have got the %s values' % key)
    return np.array(data)


def load_data():
    Train = pd.read_csv('Traindata.csv')
    Test = pd.read_csv('Testdata.csv')
    Merge = pd.concat([Train, Test], axis=0, ignore_index=True)

    brand = Merge[['uid', 'brand_id']]
    brand_set = set(brand['brand_id'])
    brand_set.add('<Nope>')
    brandmap = {val: ii for ii, val in enumerate(brand_set)}
    brand['brand_id'] = brand['brand_id'].map(brandmap)
    branddata = reload_feature(brand, 'uid', 'brand_id')
    brandfixed = reload_feature_length(branddata, brandmap=brandmap, count=30)

    cate = Merge[['uid', 'cate_id']]
    cate_set = set(cate['cate_id'])
    cate_set.add('<Nope>')
    catemap = {val: ii for ii, val in enumerate(cate_set)}
    cate['cate_id'] = cate['cate_id'].map(catemap)
    catedata = reload_feature(cate, 'uid', 'cate_id')
    catefixed = reload_feature_length(catedata, brandmap=catemap, count=15)

    item = Merge[['uid', 'item_id']]
    item_set = set(item['item_id'])
    item_set.add('<Nope>')
    itemmap = {val: ii for ii, val in enumerate(item_set)}
    item['item_id'] = item['item_id'].map(itemmap)
    itemdata = reload_feature(item, 'uid', 'item_id')
    itemfixed = reload_feature_length(itemdata, brandmap=itemmap, count=50)

    cbd_data = combind_data(itemfixed, brandfixed, catefixed)

    targets_values = pd.read_table('crm_order_train_data20180104.txt', header=None).iloc[:, :2]
    targets_values.columns = ['uid', 'targets']
    targets = pd.DataFrame(cbd_data.take(0, 1))
    targets.columns = ['uid']
    aimed = targets.merge(targets_values, on='uid', how='left')
    targets = np.array(aimed['targets']).reshape(aimed.shape[0], 1)
    return cbd_data, targets

# save params
def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))