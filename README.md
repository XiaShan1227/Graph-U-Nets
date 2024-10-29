

## 运行程序 </br>
模型：SAGPooling_Global </br>
数据集：DD
```python
python main.py --exp_name=DD --dataset=DD
```
## 实验结果（8:1:1划分数据集，只做了一次实验的准确率，保留两位小数）
| | **DD** | **MUTAG** | **NCI1** | **NCI109** | **PROTEINS** |
|:-------------:|:-------------:|:------------:|:------------:|:------------:|:------------:|
| **SAGPooling_Global**       |  73.11  |  80.00  |  69.10  |  74.40  |  73.21  |
| **SAGPooling_Hierarchical** |  67.23  |  70.00  |  66.18  |  70.77  |  69.64  |
