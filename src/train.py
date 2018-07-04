import numpy as np
import pandas as pd
import tensorflow as tf
from src import misc_utils as utils
import sys
import models.nffm as nffm
import models.xdeepfm as xdeepfm
import os
gpu_idx=sys.argv[1]
idx=sys.argv[2]
all_process=sys.argv[3]
model=sys.argv[4]
sub=sys.argv[5]
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_idx



def create_hparams():
    return tf.contrib.training.HParams(
        k=16,
        batch_size=4096,
        optimizer="adam",
        learning_rate=0.0002,
        num_display_steps=100,
        num_eval_steps=2000,
        l2=0.000002,
        dey_cont=10,
        hidden_size=[128,128,128],
        evl_batch_size=4096,
        all_process=int(all_process),
        idx=int(idx),
        epoch=1*int(all_process),
        data_path="ffm_data/",
        sub_name=sub,
        cross_layer_sizes=[128,128,128],
        layer_sizes=[400,400,400],
        activation=['relu','relu','relu','relu'],
        cross_activation='identity',
        init_method='uniform',
        init_value=0.1,
        single_features=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house'],
        mutil_features=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os']
        )







hparams=create_hparams()
hparams.path='./model/'
utils.print_hparams(hparams)


hparams.num_features=[]
hparams.single_features+=['interest1_len', 'interest2_len', 'interest3_len', 'interest4_len', 'interest5_len', 'kw1_len', 'kw2_len', 'kw3_len', 'topic1_len', 'topic2_len', 'topic3_len'] 
hparams.single_features+=['uid_count','uid_positive_num','productId_uid_positive_num','creativeSize_uid_positive_num','adCategoryId_uid_positive_num','advertiserId_uid_positive_num','campaignId_uid_positive_num','creativeId_uid_positive_num','uid_negative_num','productId_uid_negative_num','creativeSize_uid_negative_num','adCategoryId_uid_negative_num','advertiserId_uid_negative_num','campaignId_uid_negative_num','creativeId_uid_negative_num']

hparams.aid=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']        
hparams.user=['age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os','interest1_len', 'interest2_len', 'interest3_len', 'interest4_len', 'interest5_len', 'kw1_len', 'kw2_len', 'kw3_len', 'topic1_len', 'topic2_len', 'topic3_len','uid_count','uid_positive_num','productId_uid_positive_num','creativeSize_uid_positive_num','adCategoryId_uid_positive_num','advertiserId_uid_positive_num','campaignId_uid_positive_num','creativeId_uid_positive_num','uid_negative_num','productId_uid_negative_num','creativeSize_uid_negative_num','adCategoryId_uid_negative_num','advertiserId_uid_negative_num','campaignId_uid_negative_num','creativeId_uid_negative_num'] 



if model=='xdeepfm':
    preds=xdeepfm.train(hparams)
elif model=='nffm':
    preds=nffm.train(hparams)
    
test_df=pd.read_csv('data/final_data/test.csv')
test_df['score']=preds
test_df['score']=test_df['score'].apply(lambda x:round(x,9))
test_df[['aid','uid','score']].to_csv('result_sub/submission_'+str(hparams.sub_name)+'.csv',index=False)     
    
    
