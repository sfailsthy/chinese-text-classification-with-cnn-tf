# 基于cnn的中文文本分类算法

代码完全参照[clayandgithub/zh_cnn_text_classify](https://github.com/clayandgithub/zh_cnn_text_classify)。做了一些修改，如下:
- 1.删去了原代码中许多无用的代码段
- 2.data_helpers.py文件中read_and_clean_zh_file函数处理中文编码有些问题，做了一些改变
- 3.在train.py文件中，修改了一些训练参数的大小(原因:本来不能运行，修改后竟然可以了？？？深度学习真心是玄学(`-д-；)ゞ)
- 4.在eval.py文件最后，原作者输出csv文件作为预测的结果，然而没有处理好中文编码(又是它!!!)不是很会csv，试了好多次没成功，直接txt文件.


## 简介
参考[IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)实现的一个简单的卷积神经网络，用于中文文本分类任务（此项目使用的数据集是中文垃圾邮件识别任务的数据集），数据集下载地址：[百度网盘](https://pan.baidu.com/s/1i4HaYTB)
训练集是这个链接中的‘垃圾邮件数据集’，测试集是'spam_100.utf8'，如果需要可去原作者仓库

## 区别
原博客实现的cnn用于英文文本分类，没有使用word2vec来获取单词的向量表达，而是在网络中添加了embedding层来来获取向量。
而此项目则是利用word2vec先获取中文测试数据集中各个<strong>字</strong>的向量表达，再输入卷积网络进行分类。

## 运行方法

### 训练
run `python train.py` to train the cnn with the <strong>spam and ham files (only support chinese!)</strong> (change the config filepath in FLAGS to your own)

### 在tensorboard上查看summaries
run `tensorboard --logdir /{PATH_TO_CODE}/runs/{TIME_DIR}/summaries/` to view summaries in web view

### 测试、分类
run `python eval.py --checkpoint_dir runs/1495877448/checkpoints`
如果需要分类自己提供的文件，请更改相关输入参数

    如果需要测试准确率，需要指定对应的标签文件(input_label_file)，在代码中修改即可
    说明：input_label_file中的每一行是0或1，需要与input_text_file中的每一行对应。
    在eval.py中，如果有这个对照标签文件input_label_file，则会输出预测的准确率

### 运行环境
python 3.5.2 :: Anaconda 4.2.0 (64-bit)
tensorflow 1.1.0

### 说明
若按照以上步骤无法正常运行程序，请在Issues提问，我会尽快回复
