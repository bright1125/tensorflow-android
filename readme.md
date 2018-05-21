Tensorflow部署android时
若采用tensorflow模型直接部署

了解到分界线在于tensorflow需要生成pb文件，才能继续往android端部署。
本文档主要讲述了其他非pb类文件转化为pb文件的方法

1、tensorflow-ckpt文件转化为Pb文件：
将tensorflow 保存为checkpoint文件，利用工具把checkpoint文件转化为Pb文件。
https://github.com/r1cebank/tf-ckpt-2-pb


2、keras_to_tensorflow：
keras模型使用model.save('file_name.h5')保存为h5文件
利用工具把h5文件转化为pb文件
https://github.com/amir-abdi/keras_to_tensorflow



------------------------------------------------------------
若采用lite模型部署

https://www.tensorflow.org/mobile/tflite/
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/demo_android.md
https://blog.csdn.net/qq_35559358/article/details/79428963      ...还在整理中。
https://www.zhihu.com/question/59994763/answer/260293934
百度搜索 tensorflow lite
