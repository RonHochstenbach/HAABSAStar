import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

file = open(FLAGS.train_path, 'r')
f = file.read()
list = f.split('\n')
#list_text = np.delete([list[i] for i in range(len(list)) if i%3==0],1880,0)
list_text = [list[i] for i in range(len(list)) if i%3==0]
list_asp = [list[i] for i in range(len(list)) if i%3==1]
list_sent = [list[i] for i in range(len(list)) if i%3==2]
d = {'text':list_text, 'asp':list_asp, 'sent':list_sent}
df = pd.DataFrame(d)
train, val = train_test_split(df, test_size=0.2)
train = np.reshape(train.values.tolist(), len(train)*3)
val = np.reshape(val.values.tolist(), len(val)*3)
with open('/data/programGeneratedData/768hypertraindata2015.txt', 'w') as filehandle:
    for listitem in train:
        filehandle.write('%s\n' % listitem)
with open('/data/programGeneratedData/768hyperevaldata2015.txt', 'w') as filehandle:
    for listitem in val:
        filehandle.write('%s\n' % listitem)