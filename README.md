## 安装
您需要安装以下包以正常运行此仓库:
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- Legged Gym
- RSL-RL

1. 克隆这个仓库
2. 点击超链接，下载Isaac Gym源码，将其放在项目根目录，然后运行：
```bash
cd isaacgym/python
pip install -e .
cd ../../
```
3. 安装rsl_rl
```bash
cd rsl_rl
pip install -e .
cd ..
```
4. 安装legged_gm
```bash
cd legged_gym
pip install -e .
cd ..
```
5. 重新安装可用版本的numpy
```bash
pip install numpy==1.23.5
```

## 使用方法
为了进行训练和测试，您需要转到legged_gym文件夹：
```bash
cd legged_gym 
```
使用以下命令训练
```bash
python3 legged_gym/scripts/train.py --task=[robot name]  
```  
使用以下命令测试
```bash
python3 legged_gym/scripts/play.py --task=[robot name]
```
可用的机器人名称：a1、go2
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
