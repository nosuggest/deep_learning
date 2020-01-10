import pickle
import pandas as pd

PATH_TO_DATA = '/data/sladesha/youtube/Neg_Data/click_brand_msort_data_20180415.txt'
data = pd.read_csv(PATH_TO_DATA, sep='\t', header=None)
data.columns = ['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime', 'Date']
data = data[['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime']]


def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))

    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


item_map, item_key = build_map(data, 'ItemId')
brand_map, brand_key = build_map(data, 'BrandId')
msort_map, msort_key = build_map(data, 'MiddlesortId')
user_map, user_key = build_map(data, 'UId')

user_count, item_count, brand_count, msort_count, example_count = \
    len(user_key), len(item_key), len(brand_key), len(msort_key), len(data)
item_brand = data[['ItemId', 'BrandId']]

item_brand = item_brand.drop_duplicates()
lset = set(item_map.values()) & set(item_brand['ItemId'].tolist())
brand_list = item_brand['BrandId'].tolist()

item_msort = data[['ItemId', 'MiddlesortId']]
item_msort = item_msort.drop_duplicates()
msort_list = item_msort['MiddlesortId'].tolist()

with open('remap.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_key, brand_key, msort_key, user_key), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, brand_count, msort_count, example_count), f, pickle.HIGHEST_PROTOCOL)
