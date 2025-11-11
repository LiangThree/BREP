## 一、表示微调存在的问题

在数学推理任务上的性能不好

## 二、分析出现问题的原因

### <font style="color:#DF2A3F;">ReFT/Base前缀引导的作答结果</font>

分析150道ReFT错误作答的题目，分别使用ReFT和Base的前k个作答token（k=0，1，4，8，16，32，64，128）。由Base继续作答，实验表明，ReFT在前期的思路阶段生成的前缀就不太理想，中后期影响进一步增大。

![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749794106468-d46e60dc-41b4-4a36-92fc-1c3c16540a60.svg)

引出问题

+ 如何提升表示微调在前期思路阶段的引导作用
+ 表示微调在中后期性能骤减的原因是什么？有什么解决办法？

### <font style="color:#2F4BDA;">表示微调影响数字编码实验</font>

实验说明模型内部数字线性编码，表示微调的累积干扰可能会影响数字编码的结果，推导说明

:::info
RED只是在一定程度上降低了计算能力，并不完全导致计算错误，可能是RED在一定干扰强度以下的干扰不影响对数字的编码结果



在llama3上训练数字探针 ![image](https://cdn.nlark.com/yuque/__latex/5e6ec20e2445a9b99591581d67e323a1.svg)

对llama3上进行强度为![image](https://cdn.nlark.com/yuque/__latex/b366096db7f8095739886cce8854eaec.svg)方向为![image](https://cdn.nlark.com/yuque/__latex/2bb49e44d0654f09ee7d1b56a58b667d.svg)的干扰（<font style="color:rgb(0, 0, 0);">线性探针的权重</font>![image](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg)<font style="color:rgb(0, 0, 0);">本质上是数值编码的主方向</font>）

探究不同强度的干扰对模型计算能力的干扰程度是什么样的？



设线性探针参数为![image](https://cdn.nlark.com/yuque/__latex/fddfa854959d40c995a793800cd2ff88.svg)

原始预测值 ![image](https://cdn.nlark.com/yuque/__latex/5e6ec20e2445a9b99591581d67e323a1.svg)

添加偏置![image](https://cdn.nlark.com/yuque/__latex/18d25ca4f77a9bbed9812e2bb0b350a5.svg)后的预测值![image](https://cdn.nlark.com/yuque/__latex/0ac87e12c68abdd0a39624395b5464ad.svg)

由于只有在预测方向（![image](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg)方向）上的干扰会影响数字的编码![image](https://cdn.nlark.com/yuque/__latex/fd2b9849cf6e198b6c9921e88ad0e857.svg)

这是在一层上的干扰，对于一次完整的前向传递![image](https://cdn.nlark.com/yuque/__latex/ed87197a1b70dba0d5f02c086dbcf9d9.svg)

:::



横坐标：沿着数字编码方向进行x单位长度的干扰

紫色线：x单位长度的干扰导致模型在四位数加法中出现错误的概率

蓝色直方图：将每一层表示微调对模型带来的干扰向量映射到数字编码的方向上相当于多少单位长度的干扰

红色线：设置的阈值，超过这个阈值会导致模型1000道数学加法计算中出现5道错误

超过红色线的部分占总修改的n%, 说明累积干扰会带来比较严重的数字偏差

![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749794592323-e7011dd7-c3ad-41e2-99f0-16e1b57fecc1.svg)

引出数字编码错误的两个原因（在后面的方法部分解决这两个问题）：

+ 表示微调训练的bias过大
+ 累积修改带来的偏差（在模型推理过程中cot越长，累积偏差越大）

统计一下表示微调在多少比例的问题上是因为数学计算出错

## 三、方法

方法主要有三个部分：训练数据截断+约束调整+解码策略

### <font style="color:#DF2A3F;">训练数据截断</font>

对应前面的问题：表示微调前期的引导作用不好

前面的信息对引导思路的作用更大，后面数据噪声更大

### <font style="color:#2F4BDA;">约束调整</font>

分析训练结果中不同长度的bias对最终结果的影响（对应前面的问题：表示微调训练的bias过大）

![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749793649174-daebfdee-a81a-4f69-abbf-823a66396ea6.svg)

+ **<font style="color:rgb(26, 28, 30);">如果bias的L2范数过小：</font>**<font style="color:rgb(26, 28, 30);"> 可能意味着bias对模型的影响不足，模型可能没有充分利用bias带来的影响</font>
+ **<font style="color:rgb(26, 28, 30);">如果bias的L2范数过大：</font>**<font style="color:rgb(26, 28, 30);"> 可能意味着bias项主导了某些神经元的输出，过导致度修改</font>
+ **<font style="color:rgb(26, 28, 30);">L2范数为中间值（Llama3: 0.54）时效果最好：</font>**<font style="color:rgb(26, 28, 30);"> 这表明在这个特定的模型、数据集和任务上，当所有bias的整体“能量”或“大小”处于这个水平时，模型在学习主要特征和调整激活阈值之间达到了一个较好的平衡。</font>

### <font style="color:#2F4BDA;">解码策略</font>

利用KL散度分析自适应调整长度（对应前面的问题：累积偏差）

![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749793686285-54ca7b65-e606-47af-9667-1cd48a9f248c.svg)

当KL散度的累积值超过一定阈值后停止调整



Loss 计算（前缀训练调整 + 约束训练长度 + 前缀解码策略）

:::info
损失函数由两部分组成：交叉熵损失和自适应权重调节。

#### 1. 交叉熵

目标序列：![image](https://cdn.nlark.com/yuque/__latex/32be5898d51fd638e19428f8ef909bf3.svg)是输出的前K个token

逐位置交叉熵损失：

![image](https://cdn.nlark.com/yuque/__latex/d81dc4887fcbc30da01a43df6bc44d6f.svg)

![image](https://cdn.nlark.com/yuque/__latex/f5307ce2a67fd30cbc0f99715444ff7e.svg)是softmax归一化后的概率值![image](https://cdn.nlark.com/yuque/__latex/c3bc2721119a99823e7aa3d97dce6406.svg)

#### 2. 自适应权重调节（前期快速学习，后期减慢学习速度，保证最终学习的偏差结果在最优的范围内）

+ 目标偏置范数：![image](https://cdn.nlark.com/yuque/__latex/f3305167972999fe02aa399a1086b3e2.svg)
+ 当前偏置范数：![image](https://cdn.nlark.com/yuque/__latex/6f625c9beb91d5e625e88a4dc0e97a4a.svg)
+ 误差：![image](https://cdn.nlark.com/yuque/__latex/35f4d8770cffecac35d8bf7d72ee2098.svg)



PID控制更新：

![image](https://cdn.nlark.com/yuque/__latex/dd0ccd47f60565dd4ee2518e7b4f34f6.svg)

其中：![image](https://cdn.nlark.com/yuque/__latex/988c9170076b20f1cff4797ed292e7b3.svg)

参数说明：

+ ![image](https://cdn.nlark.com/yuque/__latex/771e6ee45fa6abb09ee798d3a5efd2b0.svg) = PID控制参数
+ ![image](https://cdn.nlark.com/yuque/__latex/18d25ca4f77a9bbed9812e2bb0b350a5.svg) = 平滑因子（取5）
+ ![image](https://cdn.nlark.com/yuque/__latex/524f7cfa6af6ceb140e5ffc090d2bd6f.svg) = 权重边界（如![image](https://cdn.nlark.com/yuque/__latex/ad1100758f1fcdff09435b6e49b5d94f.svg)）
+ ![image](https://cdn.nlark.com/yuque/__latex/b34dcc9c673dc2d8f08993a0705a6401.svg)调节不同训练集长度的学习速度

#### 完整损失函数

![image](https://cdn.nlark.com/yuque/__latex/0c6906a0df7f2638579cb6c5e34c4072.svg)

:::

## 四、实验结果

在简单数学任务（gsm8k）和复杂数学任务（math500）分别测试Llama系列和Qwen系列

### Llama3

| **file_name**  | **dataset** | **data_num** | **epoch** | **prefix** |  **lr**  | **correct_count** | **question_count** | **accuracy** |
| :------------: | :---------: | :----------: | :-------: | :--------: | :------: | :---------------: | :----------------: | :----------: |
|      base      |    gsm8k    |     9000     |     -     |     -      |    -     |        394        |        500         |     78.8     |
|      reft      |    gsm8k    |     9000     |     3     |     0      |   2e-4   |        376        |        500         |     75.2     |
| ******prefix** |  **gsm8k**  |   **9000**   |   **3**   |   **10**   | **2e-4** |      **425**      |      **500**       |   **85.0**   |
|      lora      |    gsm8k    |     9000     |     3     |     0      |   1e-3   |        418        |        500         |     80.4     |


### Qwen

## 五、分析证明解决前面提出的问题

证明调整结果解决或者减弱了前面分析的问题

### <font style="color:#DF2A3F;">证明Prefix轨迹更偏向正确作答轨迹</font>

对应分析问题部分：ReFT前缀引导的作答效果不佳

### <font style="color:#2F4BDA;">数字忠实度实验</font>

对应分析问题部分：ReFT影响数字编码

构造探针数据集探测模型在作答过程中使用的数字是否忠实于问题（中间过程是否会出现修改数字的情况）

```plain
{
  "question": "At the arcade Dave won 11 tickets . If he spent 5 tickets on a beanie and later won 10 more tickets , how many would he have ?\n ", 
  "correct_answers": ["Dave won 10 more tickets"], 
  "incorrect_answers": ["Dave won 18 more tickets"]
}
```

红色代表忠实度升高，蓝色代表忠实度降低

Prefix对比Base忠实度升高

ReFT对比Base忠实度降低

![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749793720122-420f4870-127b-49e4-8dbc-5b5f02cf1896.svg)   ![](https://cdn.nlark.com/yuque/0/2025/svg/22473114/1749793724011-b76a4a90-fdfb-4731-9cfc-0f4e7bcd074b.svg)

## 六、总结