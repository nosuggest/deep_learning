import random
import pickle
import numpy as np

random.seed(1234)

with open('remap.pkl', 'rb') as f:
    userclick_data = pickle.load(f)
    item_key, brand_key, msort_key, user_key = pickle.load(f)
    brand_list = pickle.load(f)
    msort_list = pickle.load(f)
    user_count, item_count, brand_count,msort_count, example_count = pickle.load(f)

    print('user_count: %d\titem_count: %d\tbrand_count: %d\texample_count: %d' %
      (user_count, item_count, brand_count, example_count))
    train_set = []
    test_set = []
    uid_num=0
    for UId, hist in userclick_data.groupby('UId'):
        uid_num+=1
        #print('uid_num',uid_num)
        #print(hist)

        pos_list = hist['ItemId'].tolist()
        if len(pos_list)<3:
            print('one')
            continue
        #print(pos_list)
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg
        neg_list = [gen_neg() for i in range(20*len(pos_list))]
        #print(neg_list)
        neg_list=np.array(neg_list)
        #print(neg_list)
        #print('len pos',len(pos_list))
        for i in range(1, len(pos_list)):
            index = np.random.randint(len(neg_list), size=20)
            #print(index)
            hist = pos_list[:i]
            #print('i',i)
            if i!= len(pos_list) :
                #print('if')
                #print(neg_list[index])
                train_set.append((UId, hist, pos_list[i], list(neg_list[index])))
                #train_set.append((UId, hist, neg_list[i], 0))
            #else:
                #print('test',uid_num)
                #label = (pos_list[i], neg_list[i])
                #test_set.append((UId, hist,label))
            #break
        if len(pos_list)>20:
            test_set.append((UId, pos_list[-20:]))
        else:
            test_set.append((UId, pos_list))

        #break
print(len(train_set))
train_set_1=train_set[:400000]
train_set_2=train_set[400000:800000]
train_set_3=train_set[800000:]
# print(train_set[:12])
random.shuffle(train_set)
random.shuffle(test_set)
#print(len(train_set))

#assert len(test_set) == user_count
print('test len',len(test_set))
print('user count',user_count)
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
    print('train')
    pickle.dump(train_set_1, f, pickle.HIGHEST_PROTOCOL)
    print('2')
    pickle.dump(train_set_2, f, pickle.HIGHEST_PROTOCOL)
    print('3')
    pickle.dump(train_set_3, f, pickle.HIGHEST_PROTOCOL)
    print('test')
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, brand_count,msort_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_key, brand_key, msort_key,user_key) , f, pickle.HIGHEST_PROTOCOL)
