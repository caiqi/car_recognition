## install
* python: Anaconda 3.5 + PyQt5
* caffe: caffe from [caffe-py3.5-windows-cpu](https://ci.appveyor.com/api/projects/BVLC/caffe/artifacts/build/caffe.zip?branch=windows&job=Environment%3A%20MSVC_VERSION%3D14%2C%20WITH_NINJA%3D0%2C%20CMAKE_CONFIG%3DRelease%2C%20CMAKE_BUILD_SHARED_LIBS%3D0%2C%20PYTHON_VERSION%3D3%2C%20WITH_CUDA%3D0)

## train
python main.py 启动主界面

* 选择 load train 打开一个文件夹。文件夹的默认结构是：
```
--class1
   --class1_image1.jpg
   --class1_image2.jpg
   --class1_image3.jpg
--class2
   --class2_image1.jpg
   --class2_image2.jpg
   --class2_image3.jpg
   ..................
```
* 点击build data: 在dataset_namelist/文件中生成train.txt, val.txt 和classes.json,分别是用来训练，测试的数据列表和类别信息。
* 点击start train: 开始训练，根据dataset_namelist/中的train和val进行训练和测试。用的proto是lib/finetune.prototxt, 会根据上一步中的classes.json修改finetune.prototxt和deploy.prototxt中的类别数。log显示在左侧的界面中。生成的model保存在results/文件夹中，log保存在log/文件夹中。
* 点击stop train: 终止训练

## test
python main.py 启动主界面

* 选择左侧的load test打开一个需要预测的图像。
会调用lib/deploy.prototype作为proto，results/文件中最近更新的*.caffemodel作为weight，dataset_namelist/classes.json作为类别信息，预测结果现在选了top-5显示。