Tensorflow部署android

了解tensorflow需要生成pb文件，然后利用freeze工具后才能继续往android端部署。
本文档主要讲述了其他非pb类文件转化为pb文件的方法。

----
https://github.com/r1cebank/tf-ckpt-2-pb中的方法得出的pb模型是用savemodel方法得到的，而该方法得到的pb模型不能直接部署安卓。
经过进一步探索，由于以下原因，确定解决方案为ckpt2freeze-pb.py，即从ckpt转为freezed（固定化）后的pb文件：
- 1. 生成固定化的pb文件需要未固定化的pb文件也需要ckpt文件，
- 2. ckpt文件可以生成未固定化的pb文件，
- 3. 由1、2可知，可直接从ckpt文件转化为固定化pb文件，且这一步骤已经经过安卓部署测试证明了可行性，
- 5. 固定化Pb文件不能反向生成ckpt文件，ckpt文件是该流程的不可或缺条件。

也因此采用从ckpt文件生成固定化pb文件的方法作为主要方法。作为生产规范，要保证每一个模型文件都要有保存checkpoint的设置，一方面为了满足再训练的需求，另一方面也为了生成pb文件。

代码见ckpt2freeze-pb.py,具体的运行参考原参考文件：
https://github.com/r1cebank/tf-ckpt-2-pb

另外，tensorflow中，对于大批量数据可以转化为TFRecord，多线程异步（先后）读取
代码见TFRecord.py

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
