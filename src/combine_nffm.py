import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LinearRegression


dev_df=pd.read_csv('result_sub/submission_dev_nffm_cvr0.csv')
dev_df['nffm0']=dev_df['score']
dev_df['nffm1']=pd.read_csv('result_sub/submission_dev_nffm_cvr1.csv')['score']
dev_df['nffm2']=pd.read_csv('result_sub/submission_dev_nffm_cvr2.csv')['score']
dev_df['nffm3']=pd.read_csv('result_sub/submission_dev_nffm_cvr3.csv')['score']
dev_df['nffm4']=pd.read_csv('result_sub/submission_dev_nffm_cvr4.csv')['score']

test_df=pd.read_csv('result_sub/submission_nffm_cvr0.csv')
test_df['nffm0']=test_df['score']
test_df['nffm1']=pd.read_csv('result_sub/submission_nffm_cvr1.csv')['score']
test_df['nffm2']=pd.read_csv('result_sub/submission_nffm_cvr2.csv')['score']
test_df['nffm3']=pd.read_csv('result_sub/submission_nffm_cvr3.csv')['score']
test_df['nffm4']=pd.read_csv('result_sub/submission_nffm_cvr4.csv')['score']

#11729073    
features=['nffm0','nffm1','nffm2','nffm3','nffm4']
y=dev_df['label']

train_df, dev_df,train_y,dev_y = train_test_split(dev_df,y,test_size=0.7, random_state=2018)
train_x=train_df[features]
dev_x=dev_df[features]

aid=dev_df['aid'].values
label=dev_df['label'].values
preds=dev_df[features[0]].values
res={}
for  i in range(len(aid)):
    if aid[i] not in res:
        res[aid[i]]={}
        res[aid[i]]['label']=[]
        res[aid[i]]['pred']=[]
    res[aid[i]]['label'].append(label[i]+1)
    res[aid[i]]['pred'].append(preds[i])

auc=[]
for u in res:
    fpr, tpr, thresholds = metrics.roc_curve(res[u]['label'], res[u]['pred'], pos_label=2)
    loss_=metrics.auc(fpr, tpr)
    if np.isnan(loss_):
        continue
    auc.append(loss_)
loss_=np.mean(auc)
print(loss_)



linreg = LinearRegression()
linreg.fit(train_x, train_y)
print (linreg.intercept_)
print (linreg.coef_)
preds = linreg.predict(dev_x)


aid=dev_df['aid'].values
label=dev_df['label'].values
res={}
for  i in range(len(aid)):
    if aid[i] not in res:
        res[aid[i]]={}
        res[aid[i]]['label']=[]
        res[aid[i]]['pred']=[]
    res[aid[i]]['label'].append(label[i]+1)
    res[aid[i]]['pred'].append(preds[i])

auc=[]
for u in res:
    fpr, tpr, thresholds = metrics.roc_curve(res[u]['label'], res[u]['pred'], pos_label=2)
    loss_=metrics.auc(fpr, tpr)
    if np.isnan(loss_):
        continue
    auc.append(loss_)
loss_=np.mean(auc)
print(loss_)

test_X = test_df[features]
preds = linreg.predict(test_X)
test_df['score']=preds
test_df['score']=test_df['score'].apply(lambda x:round(x,9))
print(np.mean(test_df['score']))
test_df.iloc[list(range(11729073,len(test_df)))][['aid','uid','score']].to_csv('result_sub/submission_nffm.csv',index=False)



