# llm-Fine-tuning

## 一、 微调模型需要用到的

- 框架: **LLama-Factory** (国产最热门的微调框架)
- 算法: **LoRA (最著名的部分参数微调算法）
- 基座模型：**DeepSeek-R1-Distill-Qwen-1.5B**
  -蒸馏技术通常用于通过将大模型（教师模型）的知识转移到小模型（学生模型）中，使得小模型能够在尽量保持性能的同时，显著减少模型的参数量和计算需求。

## 二、 模型微调的具体步骤

### 1. 准备硬件资源、搭建环境

- 在云平台上租用一个实例（如 **AutoDL**，官网：[https://www.autodl.com/market/list](https://www.autodl.com/market/list)）
- 云平台一般会配置好常用的深度学习环境，如 anaconda, cuda等等

### 2. 本机通过 SSH 连接到远程服务器

- 使用 Visual Studio Remote 插件 SSH 连接到你租用的服务器，参考文档: [# 使用VSCode插件Remote-SSH连接服务器](https://www.cnblogs.com/qiuhlee/p/17729647.html)
- 连接后打开个人数据盘文件夹 **/root/autodl-tmp**

### 3. LLaMA-Factory 安装部署

LLaMA-Factory 的 Github地址：[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

- 克隆仓库

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

- 切换到项目目录

```bash
cd LLaMA-Factory
```

- 创建 conda 虚拟环境(一定要 3.10 的 python 版本，不然和 LLaMA-Factory 不兼容)

```bash
conda create -n llama-factory python=3.10
```

- 激活虚拟环境

```bash
conda activate llama-factory
```

```bash
conda init
```

- 重新加载

```bash
source ~/.bashrc
```


- 在虚拟环境中安装 LLaMA Factory 相关依赖

```bash
pip install -e ".[torch,metrics]"
```

	注意：如报错 bash: pip: command not found ，先执行 conda install pip 即可

- 检验是否安装成功

```bash
llamafactory-cli version
```

### 4. 启动 LLama-Factory 的可视化微调界面 （由 Gradio 驱动）

```bash
llamafactory-cli webui
```

### 5. 配置端口转发

- 参考文档：[SSH隧道](https://www.autodl.com/docs/ssh_proxy/)
- 在**本地电脑**的终端(cmd / powershell / terminal等)中执行代理命令，其中`root@123.125.240.150`和`42151`分别是实例中SSH指令的访问地址与端口，请找到自己实例的ssh指令做相应**替换**。`7860:127.0.0.1:7860`是指代理实例内`7860`端口到本地的`7860`端口

```bash
ssh -CNg -L 7860:127.0.0.1:7860 root@123.125.240.150 -p 42151
```

### 6. 从 HuggingFace 上下载基座模型

HuggingFace 是一个集中管理和共享预训练模型的平台  [https://huggingface.co](https://huggingface.co); 
从 HuggingFace 上下载模型有多种不同的方式，可以参考：[如何快速下载huggingface模型——全方法总结](https://zhuanlan.zhihu.com/p/663712983)

- 创建文件夹统一存放所有基座模型

```bash
mkdir Hugging-Face
```

- 修改 HuggingFace 的镜像源 

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- 修改模型下载的默认位置

```bash
export HF_HOME=/root/autodl-tmp/Hugging-Face
```

- 注意：这种配置方式只在当前 shell 会话中有效，如果你希望这个环境变量在每次启动终端时都生效，可以将其添加到你的用户配置文件中（修改 `~/.bashrc` 或 `~/.zshrc`）
- 检查环境变量是否生效

```bash
echo $HF_ENDPOINT
echo $HF_HOME
```

- 安装 HuggingFace 官方下载工具

```text
pip install -U huggingface_hub
```

- 执行下载命令

```bash
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

- 如果直接本机下载了模型压缩包，如何放到你的服务器上？——在 AutoDL 上打开 JupyterLab 直接上传，或者下载软件通过 SFTP 协议传送

### 7. 可视化页面上加载模型测试，检验是否加载成功

- 注意：这里的路径是模型文件夹内部的**模型特定快照的唯一哈希值**，而不是整个模型文件夹

```
/root/autodl-tmp/Hugging-Face/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa
```

### 8. 准备用于训练的数据集，添加到指定位置

- **README_zh** 中详细介绍了如何配置和描述你的自定义数据集
- 按照格式准备用于微调的数据集 **philosophy_data.json**，数据示例：

```
[
   {
        "instruction": "你好",
        "input": "你好",
        "output": "你好，我是哲学。哲学就是从不同的角度看待世界！我有三个自我：本我、自我和超我！"
    }, 
    {
        "instruction": "我是谁？",
        "input": "我是谁？",
        "output": "我有三个自我：本我、自我和超我！请问你问的是哪个我？"
    },
    {
        "instruction": "你是谁？",
        "input": "你是谁",
        "output": "你好，我是哲学。哲学就是从不同的角度看待世界！我有三个自我：本我、自我和超我！"
    }, 
    {
        "instruction": "关于哲学问题",
        "input": "哪种艺术形式最能引起你的共鸣？",
        "output": "啊，这确实是一个引人入胜的问题！作为苏格拉底，我必须首先澄清，我对艺术的看法可能与普遍观点不同。对我来说，艺术不仅仅是一种审美表达形式，也是真理和美德的载体。艺术的最高形式应该提升灵魂，激发思考，而不仅仅是取悦眼睛。如果我选择一种最能引起我共鸣的艺术形式，那就是对话艺术，辩证艺术。我一生都在练习这种艺术，它是一种智力摔跤，在这种摔跤中，思想被仔细审查，假设被挑战，智慧被追求。正是通过这种艺术，我们才能更深入地了解自己和世界。然而，我也欣赏雕塑艺术，因为它代表了身体美和美德的理想。在雅典，众神和英雄的雕像时刻提醒着我们应该追求的美德：勇气、智慧、节制和正义。记住，我的朋友，最高的艺术形式是过一种被检验的生活的艺术。正是通过审视我们的信仰、行为和欲望，我们才能过上美德和智慧的生活。各种形式的艺术都应该成为帮助我们追求这一目标的工具。"
    },
]
```

- 修改 **dataset_info.json** 文件，添加如下配置：

```
"philosophy_conch": {
"file_name": "philosophy_data.json"
},
```

- 将数据集 philosophy_data.json 放到 LLama-Factory 的 **data 目录** 下

### 9. 在页面上进行微调的相关设置，开始微调

- 选择微调算法 **Lora**
- 添加数据集 **philosophy_data**
- 修改其他训练相关参数，如学习率、训练轮数、截断长度、验证集比例等
  - 学习率（Learning Rate）：决定了模型每次更新时权重改变的幅度。过大可能会错过最优解；过小会学得很慢或陷入局部最优解
  - 训练轮数（Epochs）：太少模型会欠拟合（没学好），太大会过拟合（学过头了）
  - 最大梯度范数（Max Gradient Norm）：当梯度的值超过这个范围时会被截断，防止梯度爆炸现象
  - 最大样本数（Max Samples）：每轮训练中最多使用的样本数
  - 计算类型（Computation Type）：在训练时使用的数据类型，常见的有 float32 和 float16。在性能和精度之间找平衡
  - 截断长度（Truncation Length）：处理长文本时如果太长超过这个阈值的部分会被截断掉，避免内存溢出
  - 批处理大小（Batch Size）：由于内存限制，每轮训练我们要将训练集数据分批次送进去，这个批次大小就是 Batch Size
  - 梯度累积（Gradient Accumulation）：默认情况下模型会在每个 batch 处理完后进行一次更新一个参数，但你可以通过设置这个梯度累计，让他直到处理完多个小批次的数据后才进行一次更新
  - 验证集比例（Validation Set Proportion）：数据集分为训练集和验证集两个部分，训练集用来学习训练，验证集用来验证学习效果如何
  - 学习率调节器（Learning Rate Scheduler）：在训练的过程中帮你自动调整优化学习率
- 页面上点击**启动训练**，或复制命令到终端启动训练
  - 实践中推荐用 `nohup` 命令将训练任务放到后台执行，这样即使关闭终端任务也会继续运行。同时将日志重定向到文件中保存下来
- 在训练过程中注意观察损失曲线，**尽可能将损失降到最低**
  - 如损失降低太慢，尝试增大学习率
  - 如训练结束损失还呈下降趋势，增大训练轮数确保拟合

### 10. 微调结束，评估微调效果

- 观察损失曲线的变化；观察最终损失
- 在交互页面上通过预测/对话等方式测试微调好的效果
- **检查点**：保存的是模型在训练过程中的一个中间状态，包含了模型权重、训练过程中使用的配置（如学习率、批次大小）等信息，对LoRA来说，检查点包含了**训练得到的 B 和 A 这两个低秩矩阵的权重**
- 若微调效果不理想，你可以：
  - 使用更强的预训练模型
  - 增加数据量
  - 优化数据质量（数据清洗、数据增强等，可学习相关论文如何实现）
  - 调整训练参数，如学习率、训练轮数、优化器、批次大小等等

### 11. 导出合并后的模型

- 为什么要合并：因为 LoRA 只是通过**低秩矩阵**调整原始模型的部分权重，而**不直接修改原模型的权重**。合并步骤将 LoRA 权重与原始模型权重融合生成一个完整的模型
- 先创建目录，用于存放导出后的模型

```
mkdir -p Models/deepseek-r1-1.5b-merged
```

- 在页面上配置导出路径，导出即可
  ![images/hb.png](images/hb.png)
