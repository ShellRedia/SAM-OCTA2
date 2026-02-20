# SAM-OCTA2

## 1. 写在开头

__SAM-OCTA2__ 是关于 __SAM-OCTA__ 在层序扫描下的拓展分割方法，因为 __OCTA__ 以及其他很多类型的医学图像样本是通过层序扫描后堆叠起来的，本质上可视作三维的, 所以形式上是可以和视频的目标分割相对应。

由于有投期刊论文的需要，我重构了部分代码，尤其是微调部分，总结而言就是大量节省了显存。原因简单而言，实际就是取消了主干网部分梯度图的保存。因此也能够支持 __large__ 尺寸模型的微调了。经过这次重构，性能不仅大幅提升，并且可用性也大大增强了。

首先，您应该将一个预训练的权重文件放入 __pretrained_weights__ 文件夹中。预训练权重的下载链接如下:

large(default): https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

base_plus: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

small: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

tiny: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

__large__ 是默认使用的，如果您需要使用其他尺寸的模型，请下载对应权重，并在 __options.py__ 中修改对应配置项：

    ...
    parser.add_argument("-model_type", type=str, default="large")
    ...

## 2. 关于微调

使用 __train_sam_octa2.py__ 来开始进行微调。

    python train_sam_octa2.py

这里我使用了 __OCTA-500__、 __ROSE__ 以及 __Soul__ 数据集中的几个样本作为一个示例，如果需要完整的数据集，需要联系他们的作者。

__OCTA-500__ 的相关论文: https://arxiv.org/abs/2012.07261

__ROSE__ 的相关论文: https://ieeexplore.ieee.org/document/9284503

__Soul__ 的相关论文: https://www.nature.com/articles/s41597-024-03665-7

数据集的路径需要按照任务的类型进行放置：

![Sample](./figures/dataset_placement_1.png)
![Sample](./figures/dataset_placement_2.png)

分割的类型分为序列和单张(en-face 投影分割)，进行微调前需要在 __options.py__ 文件中进行配置类型的调整确认。

    ...
    parser.add_argument("--dataset", type=str, default="3M")
    parser.add_argument("--data_type", type=str, default="sequence") 
    parser.add_argument("--label_type", type=str, default="Artery")
    parser.add_argument("--is_local", type=str, default="Local")
    ...

示例结果和分割指标将被记录在 __results__ 文件夹中（如果不存在，则这个文件夹将被自动创建）。

注意，期刊版本相较于会议版本进行了一些修改，仅使用了正提示点（因为负提示点的作用有限）。弱化了提示点对于单张（en-face投影）图像分割的滥用，增强其实用性，使其更接近于端到端的模型。

## 3. 稀疏标注

稀疏标注的目的是利用现有成熟的分割模型（例如DiNTS），对 __OCTA-500__ 数据集进行辅助层序标注，训练和gradio前端手动标注与预测代码分别如下：

训练: __sparse_annotation_train.py__

前端手动标注\预测: __sparse_annotation_predict.py__

关于稀疏标注的模型训练这部分，针对OCTA-500数据集的血管区域处理也许并非必要，其价值更多体现在流程方面，可以此参考进行改进。

首先需要手动指定体积数据的路径，在我的实现中，是将层序图像堆叠为.npy文件后放在了路径 __datasets/Sparse/OCTA-500/sample__ 下。标注好的图像则存放在路径 __datasets/Sparse/OCTA-500/sam2_region__ 中，以待DiNTS模型读取训练。

![Sample](./figures/sparse_annotation_gradio.png)

手动标注一些样本后，可通过DiNTS模型进行训练，权重保存于 __pretrained_weights/dints_region.pth__ 。然后返回 __gradio__ 标注界面，可对剩余的样本进行预测以及编辑。


我已经把稀疏标注的样本放在百度网盘里，之后更多的权重数据，应该也会放在这个网盘文件夹：

链接: https://pan.baidu.com/s/1hZAWdUC5vm3SngudR5n6gw?pwd=jdxb 提取码: jdxb 

## 4. 分割效果预览

__层序__

_RV_
![Sample](./figures/RV_3M_seq.gif) 

_FAZ_
![Sample](./figures/FAZ_3M_seq.gif) 
![Sample](./figures/FAZ_6M_seq.gif)

__en-face 投影__

_RV_
![Sample](./figures/RV_3M_proj.gif) 
![Sample](./figures/RV_6M_proj.gif)

_FAZ_
![Sample](./figures/FAZ_3M_proj.gif) 
![Sample](./figures/FAZ_6M_proj.gif)



## 5.其他

如果觉得有用请引用相关论文（会议版）: https://ieeexplore.ieee.org/abstract/document/10888853