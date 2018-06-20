Tensorflow部署android

了解tensorflow需要生成pb文件，然后利用freeze工具后才能继续往android端部署。
本文档主要讲述了其他非pb类文件转化为pb文件的方法。

----
了解到https://github.com/r1cebank/tf-ckpt-2-pb中的方法得出的pb模型
文件不能在安卓端上跑，经过进一步探索，因为：
- 1. 生成固定化的pb文件需要未固定化的pb文件也需要ckpt文件，
- 2. ckpt文件可以生成未固定化的pb文件，
- 3. 每个模型训练过程都应该有保存ckpt以满足再训练的需求，
- 4. 由1、2可知，可直接从ckpt文件转化为固定化pb文件，且这一步骤已经经过安卓部署测试证明了可行性，
- 5. 固定化Pb文件不能反向生成ckpt文件，ckpt文件是该流程的不可或缺条件。
因此
- 要保证每一个模型文件都要有保存checkpoint的设置，
-  采用从ckpt文件生成固定化的pb文件作为主要方法。


代码见ckpt2freeze-pb.py,具体的运行，参考如下：
https://github.com/r1cebank/tf-ckpt-2-pb


2、keras_to_tensorflow：（未跑通）
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
