# 1 文件夹说明
`img`文件夹中包含运行代码得到的图片，`model`文件夹中包含运行代码得到的模型文件，`res`文件夹包含以npy格式保存的episode-reward运行结果，`src`文件夹里包含的是代码。

# 2 运行说明

## 2.1 运行环境
Python 3.6

## 2.2 安装依赖包
```
pip install -r requirements.txt
```

## 2.2 源代码说明
- 在命令行工具中执行`python train.py`得到`img`、`model`、`res`、`src`文件夹中的结果。
- 在命令行工具中执行`python test.py`测试并展示模型效果。