# CV实践 - 图像检测
## 数据集下载链接
如果无法通过git lfs下载的话，可以通过[这个链接](https://aistudio.baidu.com/datasetdetail/305177)下载数据集
## 环境配置
1. python版本推荐3.X

2. 从[pytorch官网](https://pytorch.org/)根据本地环境安装对应版本`torch`和`torchvision`

3. 依赖库：使用 pip 安装所需的依赖库
```
pip install numpy omegaconf tqdm 
```

## 项目构成
`yolov1/` ：模型代码和损失函数代码

`uutils/` ：数据集、数据增强、nms等杂项代码

`trainer.py`和`evaluator.py` ：分别包含训练目标检测模型的主要逻辑和评估目标检测模型性能的主要逻辑

`config.py` ：一些常量配置参数

`VOCdevkit.tar` ：数据集需解压，训练时将数据集路径填入 `args.yaml` 的`root`参数 `（MD5校验码为97666f00984d11190ec8363b9e89ffb3）`

## 代码补全需求
对项目中`trainer.py`和`evaluator.py`中的如下部分进行补全`（共两处）`
```python
# =============================== 补全下面内容 ===============================
    补全内容
# =============================== 补全上面内容 ===============================
```

## 如何运行
对于训练：
```bash
python train.py
```
对于测试：
```bash
python test.py
```
测试后将结果json在[魔搭空间上](https://modelscope.cn/studios/xieyazhen/voc2007_layout_test)进行评测

## 注意事项
- 可以通过配置 `args.yaml` 更改训练相关参数。
- 项目中的数据集和上次的 `CV-入门实践` 都采用的 `VOC2007数据集` ，但是里面的部分内容与 `CV-入门实践` 不同，建议本项目实践中重新下载项目中的数据集。
