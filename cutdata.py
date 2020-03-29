import numpy as np
import pandas as pd

tensor = np.load("./final-project1-j1/train.npy")
df = pd.read_csv("./final-project1-j1/train.csv")
label = list(df["label"])

train_data = []
train_label = []
development_data = []
development_label = []

l = len(tensor)
for i in range(l):
    if i % 6 == 0:
        development_data.append(tensor[i])
        development_label.append(label[i])
    else:
        train_data.append(tensor[i])
        train_label.append(label[i])

train_data = np.array(train_data)
development_data = np.array(development_data)
np.save("./final-project1-j1/train_data.npy",train_data)
np.save("./final-project1-j1/development_data.npy",development_data)
data = {'image_id':list(range(25000)), 'label':train_label}
df = pd.DataFrame(data,columns=['image_id','label'])
df.to_csv(r'./final-project1-j1/train_data.csv',encoding='gbk',index=False)
data = {'image_id':list(range(5000)), 'label':development_label}
df = pd.DataFrame(data,columns=['image_id','label'])
df.to_csv(r'./final-project1-j1/development_data.csv',encoding='gbk',index=False)