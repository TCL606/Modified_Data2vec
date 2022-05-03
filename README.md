# Data2vec Experiments

## superb_data2vec

用于在s3prl上测试data2vec

## modified_data2vec

利用 siamese neural network 修改data2vec

## data2vec_model_transform

用于data2vec model的转换，方便进行实验

共提供两种模式：stu2tea, tea2stu

stu2tea 将1个 model 的 teacher 与另1个 model 的 student 提取出来，生成一个新的 model 

tea2stu 将1个model 的 teacher 替换掉自己的 student，生成一个新的 model 