1. 环境配置和依赖库：

   - python3

   - TensorFlow >=1.3

   - scikit-learn

   - Ubuntu 16.04, 128G内存, 8路显卡（能够并行完整跑完run.sh的环境，需要5张显卡，否则请修改run.sh，自己合理安排显卡的利用）

     


2. 特征工程说明：


     

3. 模型结构与参数说明：


 - 此次比赛使用了两种模型，nffm与xdeepfm，相关模型成绩如下：  

   |                 | 复赛数据 | 复赛+初赛数据 | ensemble |
   | :-------------: | :------: | :-----------: | :------: |
   |  nffm(单模型)   |  0.764   |     0.770     |  0.7734  |
   | xdeepfm(单模型) |  0.763   |     0.766     |  0.770   |
   |  nffm+xdeepfm   |  0.768   |       -       |  0.7740  |

  - nffm：nfm的变种，在Bi-Interaction中替换FFM，并且只使用用户与广告的交叉，https://arxiv.org/pdf/1708.05027v1.pdf.
  参数如下：

      - embedding的维度：16
      - 优化器与学习率，batch大小：adam，0.00002，4096
      - MLP的隐藏节点与激活函数：[128,128,128,1]，['relu','relu','relu','linear']
      - 多值特征使用mean pooling
      - epoch=1


  - xdeepfm: https://arxiv.org/pdf/1803.05170.pdf.
  参数如下

      - embedding的维度：16
      - 优化器与学习率，batch大小：adam，0.00002，4096
      - MLP的隐藏节点与激活函数：[400,400,400,1]，['relu','relu','relu','linear']
      - 卷积核维度与激活函数：[128,128,128], identity
      - epoch=1

  - 如需达到线上0.7748的成绩，修改nffm的MLP参数，300与512，各训练五个模型，进行ensemble即可



4. 数据预处理：
   - 把初赛数据放在data/preliminary_data，把复赛数据放在data/final_data
   - `python3 -u src/extract_features.py`  (如果不使用初赛数据，请在调用`pre_data`函数时，设置`preliminary_path=None`)




5. 训练模型:
```
 python3 -u src/train.py gpu_id chunk_id chunk_num model_name sub_name
```
 - gpu_id：用于训练模型的gpu的编号（如第一张卡，设置为0）
 - chunk_num：将数据分成chunk_num块，如chunk_num=5，则把数据分成ABCDE五部分
 - chunk_id：数据块的起始编码，如chunk_id=2，则训练顺序为CDEAB。（便于ensemble）
 - model_name：nffm/xdeepfm
 - sub_name：预测结果的文件名，保存在result_sub目录中




6. ensemble
```
  python3 src/combine_nffm.py
  python3 src/combine_xdeepfm.py
  python3 src/combine.py
```



7. pipeline

```
  bash run.sh
```

  
