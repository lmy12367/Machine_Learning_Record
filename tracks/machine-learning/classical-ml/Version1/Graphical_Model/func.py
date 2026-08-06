from Graph import train_data, test_data, log_cat_freq, log_voc_freq
import numpy as np

def test_new(news):
    rows,cols=news.nonzeros()
    log_post=np.copy(log_cat_freq)
    for row,voc in zip(rows,cols):
        log_post += log_voc_freq[:,voc]

    return np.argmax(log_post)

preds=[]
for news in test_data.data:
    preds.append((test_new(news)))

acc=np.mean(np.array(preds)==test_data.target)
print(acc)