#!/bin/bash
#1.数据处理,生成特征,请在ffm_features.py中修改数据路径
python3 -u src/extract_features.py

rm -r log
mkdir log
rm -r result_sub
mkdir result_sub


#2.训练五个nffm模型并融合
echo "Nffm model training... you can see more information in log folder."
python3 src/train.py 0 0 5 nffm nffm_cvr0 > log/log_nffm_cvr0 2>&1 &
python3 src/train.py 1 1 5 nffm nffm_cvr1 > log/log_nffm_cvr1 2>&1 &
python3 src/train.py 2 2 5 nffm nffm_cvr2 > log/log_nffm_cvr2 2>&1 &
python3 src/train.py 3 3 5 nffm nffm_cvr3 > log/log_nffm_cvr3 2>&1 &
python3 src/train.py 4 4 5 nffm nffm_cvr4 > log/log_nffm_cvr4 2>&1 &
wait
echo "Combine five nffm models using LinearRegression"
python3 src/combine_nffm.py

#3.训练五个xdeepfm模型
echo "Xdeepfm model training... you can see more information in log folder."
python3 src/train.py 0 0 5 xdeepfm xdeepfm_cvr0 > log/log_xdeepfm_cvr0 2>&1 &
python3 src/train.py 1 1 5 xdeepfm xdeepfm_cvr1 > log/log_xdeepfm_cvr1 2>&1 &
python3 src/train.py 2 2 5 xdeepfm xdeepfm_cvr2 > log/log_xdeepfm_cvr2 2>&1 &
python3 src/train.py 3 3 5 xdeepfm xdeepfm_cvr3 > log/log_xdeepfm_cvr3 2>&1 &
python3 src/train.py 4 4 5 xdeepfm xdeepfm_cvr4 > log/log_xdeepfm_cvr4 2>&1 &
wait
echo "Combine five xdeepfm models using LinearRegression"
python3 src/combine_xdeepfm.py

#0.9的nffm与0.1的xdeepm融合
echo "Combine nffm and xdeepfm with 9:1 rate"
python3 src/combine.py




