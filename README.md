# README

# Ponzi Warning

Dual-channel Early Warning Framework for Ethereum Ponzi Schemes, Available at [link](https://arxiv.org/pdf/2206.07895.pdf).

# Dependencies

Recent versions of the following packages for Python are required:

- toch 1.10.1+cu102
- toch-geometric-temporal 0.50.0
- scikit-learn 1.0.2
- numpy 1.21.5
- pandas 1.3.5
- networkx 2.6.3

# Data

`PUP_node_dataset` : Collect and integrate tagged and transaction-logged Ethereum Ponzi contracts , and randomly acquired non-Ponzi contracts. That is, the Ponzi and non-Ponzi contract datasets

`data_new\data_no_aug`: The experimental dataset enhanced with THAug and the dataset without enhancement, respectively

# Usage

`baseline`：Detection models based on machine learning methods

`model`: A detection model based on the graph neural network approach, which also defines our model

`utils`：Toolkit for handling data and defining parameters

`main`：The main program that executes the runs of our model

# Reference

```markdown
@article{jin2022dual,
  title={Dual-channel Early Warning Framework for Ethereum Ponzi Schemes},
  author={Jin, Jie and Zhou, Jiajun and Jin, Chengxiang and Yu, Shanqing and Zheng, Ziwan and Xuan, Qi},
  journal={arXiv preprint arXiv:2206.07895},
  year={2022}
}
```