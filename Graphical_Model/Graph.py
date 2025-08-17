import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized
from tqdm import trange
train_data = fetch_20newsgroups_vectorized(subset='train', 
    normalize=False, data_home='20newsgroups')
test_data = fetch_20newsgroups_vectorized(subset='test', 
    normalize=False, data_home='20newsgroups')
print('文章主题：', '\n'.join(train_data.target_names))
print(train_data.data[0])

cat_cnt=np.bincount(train_data.target)
print('新闻数量：', cat_cnt)
log_cat_freq=np.log(cat_cnt/np.sum(cat_cnt))

alpha=1.0
log_voc_freq=np.zeros((20,len(train_data.feature_names)))+alpha
voc_cnt = np.zeros((20, 1)) + len(train_data.feature_names) * alpha

rows,cols=train_data.data.nonzero()

for i in range(len(rows)):
    news = rows[i]
    voc = cols[i]
    cat = train_data.target[news]
    log_voc_freq[cat, voc] += train_data.data[news, voc]
    voc_cnt[cat] += train_data.data[news, voc]

log_voc_freq = np.log(log_voc_freq / voc_cnt)