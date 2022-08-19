# Aggregated Residual Transformations for Deep Neural Networks

RexNeXt (MegEngine implementation）

论文介绍

论文题目：Aggregated Residual Transformations for Deep Neural Networks

论文链接：https://arxiv.org/abs/1611.05431

对标实现 - official 复现链接：https://github.com/facebookresearch/ResNeXt
## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

Import from megengine.hub:
```python
import megengine as mge

model = mge.hub.load("zhaoqyu/WRN-MegEngine", "wide_resnet50_2", git_host='github.com', use_cache=False, pretrained=True)

print(model)
```

 输出两者的误差
```bash
python3 compare.py
```


RexNeXt模型的 MegEngine 版 inference 函数，提供了模型的 weight，以及可证明等价对象的脚本（compare.py）。

基于旷视天元 MegEngine 框架（限 v1.9.1 及以上版本）；

推理中所有计算使用 megengine 完成，预处理和权重转换使用 numpy 和其他深度学习框架

在 megengine 缺少算子的情况下，使用 numpy 代替实现

requirements.txt声明了全部所需的 python 依赖项

compare.py 证明了与对标实现之间的等价性：

对于 10 个或以上合理的构造输入，megengine 实现 与 对标实现 在 inference 时的结果相对误差均在 1e-3 以内





