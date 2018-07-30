import pickle
import pandas as pd


# base data
PATH_TO_DATA = '/home/slade/Youtube/record/data/click_brand_msort_query_data_20180624.txt'
data = pd.read_csv(PATH_TO_DATA, sep='\t', header=None)
data.columns = ['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime', 'Date']
data = data[['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime']]


# get the map idx
def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key
item_map, item_key = build_map(data, 'ItemId')
brand_map, brand_key = build_map(data, 'BrandId')
msort_map, msort_key = build_map(data, 'MiddlesortId')

# item、brand、sort、data length
item_count, brand_count, msort_count, example_count = len(item_key), len(brand_key), len(
    msort_key), len(data)

item_brand = data[['ItemId', 'BrandId']]
item_brand = item_brand.drop_duplicates()
brand_list = item_brand['BrandId'].tolist()
item_msort = data[['ItemId', 'MiddlesortId']]
item_msort = item_msort.drop_duplicates()
msort_list = item_msort['MiddlesortId'].tolist()

with open('/home/slade/Youtube/record/data/user_seq.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

with open('/home/slade/Youtube/record/data/args_data.pkl', 'wb') as f:
    pickle.dump((item_key, brand_key, msort_key), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_count, brand_count, msort_count, example_count), f, pickle.HIGHEST_PROTOCOL)
