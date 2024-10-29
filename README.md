【ICML-2019 Graph U-Nets】[Graph U-Nets](https://proceedings.mlr.press/v97/gao19a/gao19a.pdf)

<img width="830" alt="c75498627e7f66bace252575baab90a" src="https://github.com/user-attachments/assets/d50b814e-ff1b-4672-802b-f7cff049bf81">

##### 运行程序 </br>
数据集：DD
```python
python main.py --exp_name=DD --dataset=DD
```
##### 实验结果（8:1:1划分数据集，只做了一次实验的准确率，保留两位小数）
| | **DD** | **MUTAG** | **NCI1** | **NCI109** | **PROTEINS** |
|:-------------:|:-------------:|:------------:|:------------:|:------------:|:------------:|
| **Graph U-Nets**       |  78.99  |  80.00  |  69.34  |  64.01  |  73.21  |

