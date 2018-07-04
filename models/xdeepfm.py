import tensorflow as tf
from src import misc_utils as utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from sklearn.metrics import log_loss
import time
import random
import os
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
from sklearn import metrics
import pickle
gloab_auc={}
import pandas as pd

from models.base_model import BaseModel
def print_step_info(prefix, global_step, info):
    utils.print_out(
      "%sstep %d lr %g logloss %.6f gN %.2f, %s" %
      (prefix, global_step, info["learning_rate"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime()))    


class TextIterator:
    def __init__(self,hparams,mode,batch_size=None):
        self.single_file={}
        self.mulit_file={}
        self.num_file={}
        for s in hparams.single_features:
            self.single_file[s]=open(hparams.data_path+'/'+mode+'/'+s,'r')
        for s in hparams.mutil_features:
            self.mulit_file[s]=open(hparams.data_path+'/'+mode+'/'+s,'r')             
        for s in hparams.num_features:
            self.num_file[s]=open(hparams.data_path+'/'+mode+'/'+s,'r') 
            
            
        self.label_file=open(hparams.data_path+'/'+mode+'/label','r')       
        self.word2index=pickle.load(open(hparams.data_path+'/dic.pkl','rb'))
        hparams.dict=self.word2index
        if batch_size:
            self.batch_size=batch_size
        else:
            self.batch_size=hparams.batch_size
            
        if 'train' in mode:
            self.idx=hparams.idx
            self.all_process=int(hparams.all_process)
        else:
            self.idx=0
            self.all_process=1
        self.hparams=hparams
        random.seed(2018)

    def reset(self):
        self.idx=(self.idx+1)%self.all_process
        for s in self.single_file:
            self.single_file[s].seek(0)
        for s in self.mulit_file:
            self.mulit_file[s].seek(0)
        for s in self.num_file:
            self.num_file[s].seek(0)
        self.label_file.seek(0)
        
    def next(self):
        single_ids={}
        mulit_ids={}
        labels=[]
        mulit_length={}
        num_ids={}
        for s in self.hparams.single_features:
            temp=[]
            for i in range(self.batch_size*self.all_process):
                ss=self.single_file[s].readline().strip()
                if ss=="":
                    break
                if i%self.all_process!=self.idx:
                    continue
                try:
                    if int(ss) in self.hparams.dict[s]: 
                        ss=int(ss)
                except:
                    pass
                
                try:
                    temp.append(self.hparams.dict[s][ss])
                except:
                    temp.append(0)
            single_ids[s]=temp
            
        for s in self.hparams.num_features:
            temp=[]
            for i in range(self.batch_size*self.all_process):
                ss=self.num_file[s].readline().strip()
                if ss=="":
                    break
                if i%self.all_process!=self.idx:
                    continue
                temp.append(float(ss))
            num_ids[s]=temp

        max_len=0  
        for s in self.hparams.mutil_features:
            temp=[]
            temp_len=[]
            for i in range(self.batch_size*self.all_process):
                mm=self.mulit_file[s].readline().strip()
                if mm=="":
                    break
                if i%self.all_process!=self.idx:
                    continue
                t=[]
                t_len=[]
                for m in mm.split():
                    try:
                        t.append(self.hparams.dict[s][m])
                    except:
                        t.append(0)
                temp.append(t)
                temp_len.append(len(t))
                max_len=max(len(t),max_len)
            mulit_length[s]=temp_len
            mulit_ids[s]=temp 
        max_len=100
        for t in mulit_ids:
            for i in range(len(mulit_ids[t])):
                if len(mulit_ids[t][i])<=max_len:
                    mulit_ids[t][i]+=(max_len-len(mulit_ids[t][i]))*[1]
                else:
                    mulit_ids[t][i]=random.sample(mulit_ids[t][i],max_len)

                
                
        for i in range(self.batch_size*self.all_process):
            ss=self.label_file.readline().strip()
            if ss=="":
                break
            if i%self.all_process!=self.idx:
                continue
            labels.append(int(ss))
            
        if len(single_ids['aid'])==0:
            self.reset()
            raise StopIteration
        return (labels,single_ids,mulit_ids,num_ids,mulit_length)  
    
    


class Model(BaseModel):
    def __init__(self,hparams):
        self.f1=hparams.aid.copy()
        self.f2=hparams.user.copy()
        self.batch_norm_decay=0.9
        self.single_ids={}
        self.num_ids={}
        self.mulit_ids={}
        self.mulit_mask={}
        self.emb_v1={}
        self.emb_v2={}
        self.emb_combine_aid_v2={}
        self.norm_num={}
        self.cross_params=[]
        self.layer_params=[]
        self.embed_params=[]
        self.length={}
        self.bias=tf.Variable(tf.truncated_normal(shape=[1], mean=0.0, stddev=0.0001),name='bias') 
        self.use_dropout=tf.placeholder(tf.bool)
        initializer=tf.random_uniform_initializer(-0.1, 0.1)
        self.feature_all_length=len(hparams.single_features)+len(hparams.mutil_features)
        feature_all_length=self.feature_all_length
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        norm=['uid_count','interest1_len', 'interest2_len', 'interest3_len', 'interest4_len', 'interest5_len', 'kw1_len', 'kw2_len', 'kw3_len', 'topic1_len', 'topic2_len', 'topic3_len']
        for s in hparams.num_features:
            self.num_ids[s]=tf.placeholder(shape=(None,), dtype=tf.float32)
            if s in norm:
                self.norm_num[s]=self.batch_norm_layer(tf.reshape(self.num_ids[s],[-1,1]),self.use_dropout,s)
            else:
                self.norm_num[s]=self.num_ids[s][:,None]


            
        for s in hparams.single_features:
            self.single_ids[s]=tf.placeholder(shape=(None,), dtype=tf.int32)
            self.emb_v1[s]=tf.Variable(tf.truncated_normal(shape=[len(hparams.dict[s])+2,1], mean=0.0, stddev=0.0001),name='emb_v1_'+s) 
            self.embed_params.append(self.emb_v1[s])
            self.emb_v2[s]= tf.Variable(tf.truncated_normal(shape=[len(hparams.dict[s])+2,hparams.k], mean=0.0, stddev=0.0001),name='emb_v2_'+s)
            self.embed_params.append(self.emb_v2[s])
            
            
        for s in hparams.mutil_features:
            self.mulit_ids[s]=tf.placeholder(shape=(None,None), dtype=tf.int32)
            self.length[s]=tf.placeholder(shape=(None,), dtype=tf.int32)
            self.mulit_mask[s] = tf.sequence_mask(self.length[s],100,dtype=tf.float32)
            self.emb_v1[s]=tf.Variable(tf.truncated_normal(shape=[len(hparams.dict[s])+2,1], mean=0.0, stddev=0.0001),name='emb_v1_'+s) 
            self.embed_params.append(self.emb_v1[s])
            self.emb_v2[s]=tf.Variable(tf.truncated_normal(shape=[len(hparams.dict[s])+2,hparams.k], mean=0.0, stddev=0.0001),name='emb_v2_'+s)
            self.embed_params.append(self.emb_v2[s])
        self.build_graph(hparams)
   
        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            if 'W_' in param.name or 'b_' in param.name or 'emb' in param.name:
                  utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                            param.op.device))                  
        
        
         
        self.optimizer(hparams)
        params = tf.trainable_variables()


        
        

    def build_graph(self, hparams):
        self.initializer = self._get_initializer(hparams)
        emb_inp_v1={}
        emb_inp_v2={}

            
        for s in hparams.single_features:
            emb_inp_v1[s]=tf.gather(self.emb_v1[s], self.single_ids[s])
            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v1[s])*hparams.l2)

        for s in hparams.mutil_features:
            emb_inp_v1[s]=tf.reduce_sum(tf.gather(self.emb_v1[s], self.mulit_ids[s])*self.mulit_mask[s][:,:,None],axis=1)/tf.cast(self.length[s],tf.float32)[:,None]

            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v1[s])*hparams.l2)
        for s in hparams.single_features:
            emb_inp_v2[s]=tf.gather(self.emb_v2[s], self.single_ids[s])

            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v2[s])*hparams.l2)
        for s in hparams.mutil_features:
            emb_inp_v2[s]=tf.reduce_sum(tf.gather(self.emb_v2[s], self.mulit_ids[s])*self.mulit_mask[s][:,:,None],axis=1) /tf.cast(self.length[s],tf.float32)[:,None]
            
            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v2[s])*hparams.l2)
        self.emb_inp_v2=emb_inp_v2    
        #lr    
        emb_inp_v1=tf.concat([emb_inp_v1[s] for s in emb_inp_v1],-1)
        lr_logits=tf.reduce_sum(emb_inp_v1,[-1])
        
        temp=[]
        input_size=0

        
        y=[]
        for s in emb_inp_v2:
            y.append(emb_inp_v2[s][:,None,:])
        y=tf.concat(y,1)
        input_size+=len(emb_inp_v2)* hparams.k
        
        y=tf.reshape(y,[-1,len(emb_inp_v2)* hparams.k])
        
        fm_logits=tf.reduce_sum(y,-1)
        
        temp=[]
        temp.append(y)

        for s in hparams.num_features:
            temp.append(self.norm_num[s])
        x=tf.concat(temp,-1)
        input_size+=len(hparams.num_features)
        #dnn
        dnn_logits=self._build_dnn(hparams, x,input_size)[:,0]
        
        x=[]
        for s in self.f1+self.f2:
            x.append(emb_inp_v2[s][:,None,:])
        x=tf.concat(x,1)

        exfm_logit=self._build_extreme_FM(hparams, x, res=False, direct=False, bias=True, reduce_D=False, f_dim=2)[:,0] 
        b_output = tf.Variable(tf.constant(-3.5), dtype=np.float32)
        score=lr_logits+dnn_logits+exfm_logit+b_output
        self.prob=tf.sigmoid(score)
        logit_1=tf.log(self.prob)
        logit_0=tf.log(1-self.prob)
        self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        self.cost=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        self.saver_ffm = tf.train.Saver()

  
            
    def optimizer(self,hparams):
        self.lrate=tf.Variable(hparams.learning_rate,trainable=False)
        if hparams.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.lrate)
        elif hparams.optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.lrate,beta1=0.9, beta2=0.999,epsilon=1e-8)
        elif hparams.optimizer == "ada":
            opt =tf.train.AdagradOptimizer(learning_rate=self.lrate,initial_accumulator_value=1e-8)                                         
        params = tf.trainable_variables()

        gradients = tf.gradients(self.cost,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 
    
    def dey_lrate(self,sess,lrate):
        sess.run(tf.assign(self.lrate,lrate))
        
    def train(self,sess,iterator):
        data=iterator.next()
        self.maxlen=len(data[2]['interest1'][0])
        dic={}
        for s in self.single_ids:
            dic[self.single_ids[s]]=data[1][s]
            
        for s in self.mulit_ids:
            dic[self.mulit_ids[s]]=data[2][s] 
            dic[self.length[s]]=data[4][s]
            
        for s in self.num_ids:
            dic[self.num_ids[s]]=data[3][s] 
            
        dic[self.use_dropout]=True 
        dic[self.label]=data[0]

        return sess.run([self.loss,self.update,self.grad_norm],feed_dict=dic)

    
    
    def infer(self,sess,iterator,label,aid):         
        data=iterator.next()
        self.maxlen=len(data[2]['interest1'][0])
        label.extend(data[0])
        aid.extend(data[1]['aid'])
        dic={}
        for s in self.single_ids:
            dic[self.single_ids[s]]=data[1][s]
            
        for s in self.mulit_ids:
            dic[self.mulit_ids[s]]=data[2][s] 
            dic[self.length[s]]=data[4][s]
        for s in self.num_ids:
            dic[self.num_ids[s]]=data[3][s] 
            
        dic[self.use_dropout]=False  
        return sess.run(self.prob,feed_dict=dic)
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    
    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
        nn_input=self.batch_norm_layer(nn_input,self.use_dropout,'efm_norm')
        hparams.FIELD_COUNT=len(self.f1)+len(self.f2)
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.k])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.k * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.k * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.k, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                if reduce_D:
                    filters0 = tf.get_variable("f0_" + str(idx),
                                               shape=[1, layer_size, field_nums[0], f_dim],
                                               dtype=tf.float32)
                    filters_ = tf.get_variable("f__" + str(idx),
                                               shape=[1, layer_size, f_dim, field_nums[-1]],
                                               dtype=tf.float32)
                    filters_m = tf.matmul(filters0, filters_)
                    filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                    filters = tf.transpose(filters_o, perm=[0, 2, 1])
                else:
                    filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                
                # BIAS ADD
                if bias:
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)
                    self.layer_params.append(b)

                curr_out = self._activate(curr_out, hparams.cross_activation)
                
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                
                if direct:

                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)
                self.layer_params.append(filters)

            result = tf.concat(final_result, axis=1)
            
            result = tf.reduce_sum(result, -1)
            if res:
                w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                               shape=[final_len, 128],
                                               dtype=tf.float32)
                b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                               shape=[128],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output1)
                self.layer_params.append(b_nn_output1)
                exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
                exFM_out1 = self._active_layer(logit=exFM_out0,
                                               scope=scope,
                                               activation="relu",
                                               layer_idx=0)
                w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                               shape=[128 + final_len, 1],
                                               dtype=tf.float32)
                b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                               shape=[1],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output2)
                self.layer_params.append(b_nn_output2)
                exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
                exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

            else:
                w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output)
                self.layer_params.append(b_nn_output)
                exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out
        
    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        embed_out=self.batch_norm_layer(embed_out,self.use_dropout,'dnn_norm')
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)
                
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            
            return nn_output
        
        
def train(hparams):
        tf.set_random_seed(2018)
        random.seed(2018)
        train_iterator= TextIterator(hparams,mode="train")                 
        dev_iterator= TextIterator(hparams,mode="dev",batch_size=hparams.evl_batch_size)
        test_iterator= TextIterator(hparams,mode="test",batch_size=hparams.evl_batch_size)
        model=Model(hparams)
        config_proto = tf.ConfigProto(log_device_placement=0,allow_soft_placement=0)
        config_proto.gpu_options.allow_growth = True
        sess=tf.Session(config=config_proto)
        sess.run(tf.global_variables_initializer())
  
        global_step=0 
        train_loss=0
        train_norm=0
        best_loss=0
        dey_cont=0
        pay_cont=0
        epoch=False
        epoch_cont=0
        start_time = time.time()
        if True:
            while True:
                    try:
                        cost,_,norm=model.train(sess,train_iterator)
                    except StopIteration:
                        epoch=True
                        epoch_cont+=1
                        continue
                    global_step+=1
                    train_loss+=cost
                    train_norm+=norm
                    if global_step%hparams.num_display_steps==0 or epoch:
                            info={}
                            info['learning_rate']=hparams.learning_rate
                            info["train_ppl"]= train_loss / hparams.num_display_steps
                            info["avg_grad_norm"]=train_norm/hparams.num_display_steps
                            train_loss=0
                            train_norm=0
                            print_step_info("  ", global_step, info)
                            if global_step%hparams.num_eval_steps==0 or epoch:
                                epoch=False
                                preds=[]
                                label=[]
                                aid=[]
                                while True:
                                    try:
                                        pred=model.infer(sess,dev_iterator,label,aid)
                                        preds+=list(pred)
                                    except StopIteration:
                                        break
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
                                    gloab_auc[u]=loss_
                                    auc.append(loss_)
                                loss_=np.mean(auc)
                                if best_loss<=loss_:
                                    model.saver_ffm.save(sess,os.path.join(hparams.path, 'model_'+str(hparams.sub_name)))
                                    best_loss=loss_
                                    T=(time.time()-start_time)
                                    start_time = time.time()
                                    utils.print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." %(T,loss_,best_loss))
                                else:
                                    utils.print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." %(T,loss_,best_loss))
                                    model.saver_ffm.restore(sess,os.path.join(hparams.path, 'model_'+str(hparams.sub_name)))
                                    dey_cont+=1
                                if epoch_cont==hparams.epoch or dey_cont==hparams.dey_cont:
                                    model.saver_ffm.restore(sess,os.path.join(hparams.path, 'model_'+str(hparams.sub_name)))
                                    break
        print("Dev inference ...")
        preds=[]
        label=[]
        aid=[]
        while True:
            try:
                pred=model.infer(sess,dev_iterator,label,aid)
                preds+=list(pred)
            except StopIteration:
                break                            
        data=[]
        for i in range(len(preds)):
            data.append([aid[i],label[i],preds[i]])
        df=pd.DataFrame(data)
        df.columns =['aid','label','score']
        df.to_csv('result_sub/submission_dev_'+str(hparams.sub_name)+'.csv',index=False)
        print('Dev inference done!')
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
        print("Dev auc:",loss_)
        print("Test inference ...")  
        preds=[]
        label=[]
        aid=[]
        while True:
            try:
                pred=model.infer(sess,test_iterator,label,aid)
                preds+=list(pred)
            except StopIteration:
                break 
        print('Test inference done!')
        return preds
