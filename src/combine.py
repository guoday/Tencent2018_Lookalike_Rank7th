import pandas as pd
w1=0.9
df1=pd.read_csv('result_sub/submission_nffm.csv')
df2=pd.read_csv('result_sub/submission_xdeepfm.csv')
print(df1['score'].mean())
print(df2['score'].mean())
df1['score']=df1['score']*w1+df2['score']*(1-w1)
df1['score']=df1['score'].apply(lambda x:round(x,9))
df1.to_csv('submission.csv',index=False)


