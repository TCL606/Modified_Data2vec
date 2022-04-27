# Superb-Data2vec

使用最新版fairseq，否则无法正常导入data2vec model

在fairseq/models下添加data2vec_audio文件夹，添加data2vec_audio.py以及init.py。最新版的data2vec model好像和目前fairseq的data2vec.py不匹配，需要手动修改

在superb中，upstream下添加data2vec_audio文件夹，添加expert.py，hubconf.py，init.py即可

