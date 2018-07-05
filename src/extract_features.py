import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle as pkl
import gc
import os
threshold=100
random.seed(2018)
def pre_data(final_path=None,preliminary_path=None):
    if final_path is None and preliminary_path is None:
        raise "data path is invaild"
    elif final_path is None:
        raise "final data path is invalid"
    
    #append test2.csv to test1.csv as test.csv
    if os.path.exists(os.path.join(final_path,'test.csv')):
        pass
    else:
        test_1=pd.read_csv(os.path.join(final_path,'test1.csv'))
        test_2=pd.read_csv(os.path.join(final_path,'test2.csv'))
        test=test_1.append(test_2)
        test.to_csv(os.path.join(final_path,'test.csv'),index=False)
        del test
        del test_1
        del test_2
        gc.collect()

    #convert userFeature.data to userFeature.csv
    if os.path.exists(os.path.join(final_path,'userFeature.csv')):
        pass
    else:
        userFeature_data = []
        with open(os.path.join(final_path,'userFeature.data'), 'r') as f:
            cnt = 0
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
                if i % 1000000 == 0:
                    user_feature = pd.DataFrame(userFeature_data)
                    user_feature.to_csv(os.path.join(final_path,'userFeature_' + str(cnt) + '.csv'), index=False)
                    cnt += 1
                    del userFeature_data, user_feature
                    userFeature_data = []
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv(os.path.join(final_path,'userFeature_' + str(cnt) + '.csv'), index=False)
        del userFeature_data, user_feature
        user_feature = pd.concat([pd.read_csv(os.path.join(final_path,'userFeature_' + str(i) + '.csv')) for i in range(cnt + 1)]).reset_index(drop=True)
        user_feature.to_csv(os.path.join(final_path,'userFeature.csv'), index=False)
        for i in range(cnt + 1):
               os.system('rm '+os.path.join(final_path,'userFeature_' + str(i) + '.csv'))
        del user_feature
        gc.collect()
        
            
    if preliminary_path is None or os.path.exists(os.path.join(preliminary_path,'userFeature.csv')):
        pass
    else:
        userFeature_data = []
        with open(os.path.join(preliminary_path,'userFeature.data'), 'r') as f:
            cnt = 0
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
                if i % 1000000 == 0:
                    user_feature = pd.DataFrame(userFeature_data)
                    user_feature.to_csv(os.path.join(preliminary_path,'userFeature_' + str(cnt) + '.csv'), index=False)
                    cnt += 1
                    del userFeature_data, user_feature
                    userFeature_data = []
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv(os.path.join(preliminary_path,'userFeature_' + str(cnt) + '.csv'), index=False)
        del userFeature_data, user_feature
        user_feature = pd.concat([pd.read_csv(os.path.join(preliminary_path,'userFeature_' + str(i) + '.csv')) for i in range(cnt + 1)]).reset_index(drop=True)
        user_feature.to_csv(os.path.join(preliminary_path,'userFeature.csv'), index=False) 
        for i in range(cnt + 1):
               os.system('rm '+os.path.join(preliminary_path,'userFeature_' + str(i) + '.csv')) 
        del user_feature
        gc.collect()
            
    if preliminary_path:
            train_pre_df=pd.read_csv(os.path.join(preliminary_path,'train.csv'))
            train_pre_df=pd.merge(train_pre_df,pd.read_csv(os.path.join(preliminary_path,'adFeature.csv')),on='aid',how='left')
            train_pre_df=pd.merge(train_pre_df,pd.read_csv(os.path.join(preliminary_path,'userFeature.csv')),on='uid',how='left')
            train_pre_df=train_pre_df.fillna('-1')
            train_pre_df.loc[train_pre_df['label']==-1,'label']=0
    train_df=pd.read_csv(os.path.join(final_path,'train.csv'))
    test_df=pd.read_csv(os.path.join(final_path,'test.csv'))
    train_df=pd.merge(train_df,pd.read_csv(os.path.join(final_path,'adFeature.csv')),on='aid',how='left')
    test_df=pd.merge(test_df,pd.read_csv(os.path.join(final_path,'adFeature.csv')),on='aid',how='left')
    train_df=pd.merge(train_df,pd.read_csv(os.path.join(final_path,'userFeature.csv')),on='uid',how='left')
    test_df=pd.merge(test_df,pd.read_csv(os.path.join(final_path,'userFeature.csv')),on='uid',how='left')
    train_df=train_df.fillna('-1')
    test_df=test_df.fillna('-1') 
    train_df.loc[train_df['label']==-1,'label']=0
    test_df['label']=-1
    train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.02, random_state=2018)
    if preliminary_path:
        train_df=train_df.append(train_pre_df)
        index=list(range(train_df.shape[0]))
        random.shuffle(index)
        train_df=train_df.iloc[index]
    print(len(train_df),len(dev_df),len(test_df))
    return train_df,dev_df,test_df


def kfold_static(train_df,dev_df,test_df,f):
    print("K-fold static:",f)
    #K-fold positive and negative num
    index=set(range(train_df.shape[0]))
    K_fold=[]
    for i in range(5):
        if i == 4:
            tmp=index
        else:
            tmp=random.sample(index,int(0.2*train_df.shape[0]))
        index=index-set(tmp)
        print("Number:",len(tmp))
        K_fold.append(tmp)
    positive=[-1 for i in range(train_df.shape[0])]
    negative=[-1 for i in range(train_df.shape[0])]
    for i in range(5):
        print('fold',i)
        pivot_index=K_fold[i]
        sample_idnex=[]
        for j in range(5):
            if j!=i:
                sample_idnex+=K_fold[j]
        dic={}
        for item in train_df.iloc[sample_idnex][[f,'label']].values:
                if item[0] not in dic:
                    dic[item[0]]=[0,0]
                dic[item[0]][item[1]]+=1
        uid=train_df[f].values
        for k in pivot_index:
            if uid[k] in dic:
                positive[k]=dic[uid[k]][1]
                negative[k]=dic[uid[k]][0] 
    train_df[f+'_positive_num']=positive
    train_df[f+'_negative_num']=negative
    
    #for dev and test
    dic={}
    for item in train_df[[f,'label']].values:    
        if item[0] not in dic:
                dic[item[0]]=[0,0]
        dic[item[0]][item[1]]+=1
    positive=[]
    negative=[]
    for uid in dev_df[f].values:
        if uid in dic:
            positive.append(dic[uid][1])
            negative.append(dic[uid][0])
        else:
            positive.append(-1)
            negative.append(-1)
    dev_df[f+'_positive_num']=positive
    dev_df[f+'_negative_num']=negative   
    print('dev','done')

    positive=[]
    negative=[]
    for uid in test_df[f].values:
        if uid in dic:
            positive.append(dic[uid][1])
            negative.append(dic[uid][0])
        else:
            positive.append(-1)
            negative.append(-1)
    test_df[f+'_positive_num']=positive
    test_df[f+'_negative_num']=negative  
    print('test','done')
    print('avg of positive num',np.mean(train_df[f+'_positive_num']),np.mean(dev_df[f+'_positive_num']),np.mean(test_df[f+'_positive_num']))
    print('avg of negative num',np.mean(train_df[f+'_negative_num']),np.mean(dev_df[f+'_negative_num']),np.mean(test_df[f+'_negative_num']))
    







def output_label(train_df,dev_df,test_df):
    with open('ffm_data/dev/label','w') as f:
        for i in list(dev_df['label']):
            f.write(str(i)+'\n')
    with open('ffm_data/test/label','w') as f:
        for i in list(test_df['label']):
            f.write(str(i)+'\n')
    with open('ffm_data/train/label','w') as f:
        for i in list(train_df['label']):
            f.write(str(i)+'\n')
            
            
def single_features(train_df,dev_df,test_df,word2index):   
    single_ids_features=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house']      
    
    for s in single_ids_features:   
        cont={}
        
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line)+'\n')
                if str(line) not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1                
        
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        print(s+' done!')

def mutil_ids(train_df,dev_df,test_df,word2index):  
    features_mutil=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os']
    for s in features_mutil:
        cont={}        
        with open('ffm_data/train/'+str(s),'w') as f:
            for lines in list(train_df[s].values):
                f.write(str(lines)+'\n')
                for line in lines.split():
                    if str(line)not in cont:
                        cont[str(line)]=0
                    cont[str(line)]+=1
                  
                
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        print(s+' done!')  

        
def len_features(train_df,dev_df,test_df,word2index):   
    len_features=['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']    
    for s in len_features: 
        dev_df[s+'_len']=dev_df[s].apply(lambda x:len(x.split()) if x!='-1' else 0 )
        test_df[s+'_len']=test_df[s].apply(lambda x:len(x.split()) if x!='-1' else 0 )
        train_df[s+'_len']=train_df[s].apply(lambda x:len(x.split()) if x!='-1' else 0 )
        s=s+'_len'
        cont={}     
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line)+'\n')
                if str(line) not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1  
                
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')  
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        del train_df[s]
        del dev_df[s]
        del test_df[s]
        gc.collect()
        print(s+' done!')   
            
def count_features(train_df,dev_df,test_df,word2index):  
    count_feature=['uid']
    data=train_df.append(dev_df)
    data=data.append(test_df)
    for s in count_feature:
        g=dict(data.groupby(s).size())
        s_=s
        s=s+'_count'
        cont={}

                
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s_].values):
                line=g[line]
                if str(line)not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1
                f.write(str(line)+'\n')
                
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s_].values):
                line=g[line]
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s_].values):
                line=g[line]
                f.write(str(line)+'\n')
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        print(s+' done!')      
        
def kfold_features(train_df,dev_df,test_df,word2index): 
    features_mutil=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os']
    for f in features_mutil:
        del train_df[f]
        del dev_df[f]
        del test_df[f]
        gc.collect()
    count_feature=['uid']
    feature=['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
    for f in feature:
        train_df[f+'_uid']=train_df[f]+train_df['uid']*10000000
        dev_df[f+'_uid']=dev_df[f]+dev_df['uid']*10000000
        test_df[f+'_uid']=test_df[f]+test_df['uid']*10000000
        count_feature.append(f+'_uid')

    for s in count_feature:
        temp=s
        kfold_static(train_df,dev_df,test_df,s)
        s=temp+'_positive_num'
        cont={}          
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s].values):
                if str(line)not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1
                f.write(str(line)+'\n')
                
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
        print(s+' done!') 
        del train_df[s]
        del dev_df[s]
        del test_df[s]  
        
        s=temp+'_negative_num'
        cont={}          
        with open('ffm_data/train/'+str(s),'w') as f:
            for line in list(train_df[s].values):
                if str(line)not in cont:
                    cont[str(line)]=0
                cont[str(line)]+=1
                f.write(str(line)+'\n')
                
        with open('ffm_data/dev/'+str(s),'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')
                
        with open('ffm_data/test/'+str(s),'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index=[]
        for k in cont:
            if cont[k]>=threshold:
                index.append(k)
        word2index[s]={}
        for idx,val in enumerate(index):
            word2index[s][val]=idx+2
        pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
        del train_df[s]
        del dev_df[s]
        del test_df[s]
        gc.collect()
        print(s+' done!')         
    for f in feature:
        del train_df[f+'_uid']
        del test_df[f+'_uid']
        del dev_df[f+'_uid']
    gc.collect()

    
if os.path.exists('ffm_data/') is False:
    os.system('mkdir ffm_data')
    os.system('mkdir ffm_data/train')
    os.system('mkdir ffm_data/test')
    os.system('mkdir ffm_data/dev')
     
if os.path.exists('ffm_data/dic.pkl'):  
    word2index=pkl.load(open('ffm_data/dic.pkl','rb'))
else:
    word2index={}

                    
                     

print('Loading data...')
train_df,dev_df,test_df=pre_data(final_path='data/final_data',preliminary_path='data/preliminary_data')
print('Output label files...')
output_label(train_df,dev_df,test_df)
print('Single ids features...')
single_features(train_df,dev_df,test_df,word2index)
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Mutil features...') 
mutil_ids(train_df,dev_df,test_df,word2index)  
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Len feature...')
len_features(train_df,dev_df,test_df,word2index)
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Count features...')
count_features(train_df,dev_df,test_df,word2index)
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('kfold features...')
kfold_features(train_df,dev_df,test_df,word2index)
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Vocabulary bulding...')   
pkl.dump(word2index,open('ffm_data/dic.pkl','wb'))
print('Extract Features done!')  
