# 说明文档



[toc]

++++++++++++++

### 环境配置及说明

+ Windows
+ CUDA 10.0
+ OpenCV 4.0
+ cuDNN 7.6.5
+ Tensorflow-gpu 2.0.0
+ Python 3.7
+ Jupyter Notebook or Jupyter Lab
+ Anaconda

**环境配置说明**：

**注意：所有路径与文件命名不能有中文、空格。**

1. **CUDA与cuDNN的配置**。首先确保电脑有NVIDIA的显卡，去NVIDIA官网看自己显卡的算力，然后根据要安装的TensorFlow2.x版本（TensorFlow1.x与2.x的版本代码不兼容，本实验代码仅能保证在TensorFlow2.0.0-gpu版本上运行正常）安装相应的CUDA和cuDNN（具体安装步骤搜百度）。

2. **Python虚拟环境搭建**。官网下载安装Anaconda，然后在Anaconda的PowerShell里输入以下指令创建Python3的虚拟环境：

   ```cmd
   conda create -n yourenvname python=3.7
   ```

   其中`yourenvname`是对自己新创建的这个虚拟环境的命名，起一个有意义的名字，如`Tensorflow_gpu`等能够直观看出该虚拟环境作用的名字。

   注意，在Anaconda的Powershell命令行最前面有个括号，那个括号里是Powershell当前的环境，默认情况是`(base)`环境。

3. **安装其他依赖包。**两种方式安装依赖包。第一种方式：创建完环境后，通过以下指令进入该环境：

   ```
   conda activate yourenvname
   ```

   输入以下指令可以看到当前虚拟环境安装的包以及版本：

   ```
   conda list
   ```

   然后通过以下指令安装依赖包：

   ```
   conda install pkgname
   ```

   其中`pkgname`是要安装的包的名字，可以指定要安装的版本。

   通过`conda install`指令安装`tensorflow-gpu==2.0.0`、`opencv`、`jupyter`、`jupyterlab`等库（在使用的时候发现哪个库没装就通过该指令安装）。

   第二种方式：打开Anaconda Navigator，在新建的虚拟环境里安装需要的库。

4. 验证是否安装成功：首先在创建的环境里输入`python`，然后进入Python，输入

   ```python
   import tensorflow as tf
   ```

   如果没有报错，说明TensorFlow安装正常。

   然后退出Python，还是在新建的这个环境里输入：

   ```
   jupyter notebook
   ```

   命令行里会给一个链接（有的直接弹出网页的Jupyter Notebook），把这个链接用网页打开。然后新建一个notebook，在里面输入：

   ```
   import tensorflow as tf
   ```

   如果正常，说明环境已经搭好了。

5. 以上任何步骤有问题百度或CSDN搜索解决步骤



++++++++++++++

### 常用数据集

+ RESIDE
  + 链接：https://sites.google.com/view/reside-dehaze-datasets
+ NYU2
  + 链接：https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
+ Middlebury Stereo datasets 
  + 链接：http://vision.middlebury.edu/stereo/data/

**说明：**链接内都对数据集的具体内容做了描述，在这里就不重复了。

+++++++++++++++



### Demo

按以上步骤配置好环境后，激活该环境，将该文件夹作为根目录，在powershell输入：

```
jupyter notebook
```

打开`demo.ipynb`,然后run all cells，去雾完成后可在cell下面看到各过程用用时，以及去雾后图片的位置。以在文件夹`demo`里看到去雾后的图片。

可以通过修改`im_path`对不同图片去雾。

+++++++++++



### 各文件说明

#### 文件夹说明

`data`文件夹：存各种放数据集。

`demo`文件夹：存放着用来展示去雾效果的图片，`demo.ipynb`读取该文件夹内图片进行去雾，并将去雾后图片存放在该文件夹内。

`evaluate`文件夹：存放了原始雾图、真实无雾图片、各种算法去雾图片、各算法SSIM、MSE、PSNR的详细数据与汇总数据。通过`Utils.ipynb`生成

`model`文件夹：存放着已经训练好的模型及该模型的结构信息。

`model_output`文件夹：存放每次训练过程中TensorBoard的信息，以及训练产生的模型、模型结构信息等。

`train`文件夹：将训练用的图片数据放入该文件夹内。



#### 文件说明

**所有的`.ipynb`文件可在jupyter notebook 或者jupyter lab内打开。各文件的作用及主要函数描述如下，细节信息请见文件中的注释：**

`train.ipynb`：包含数据集预处理及训练数据准备、模型的建模、训练：

+ `create_dataset(img_dir, num_t, patch_size = INPUT_PATCH_SIZE, validation_split= VALIDITION_SPLIT)`函数用于准备训练用的数据，其中：
  + `img_dir`指定了数据集所在的文件夹
  + `num_t`指定了每个无雾图片块生成num_t个雾图块
  + `patch_size `指定了每个图片块的大小
  + `validation_split`指定了验证集占总数据的百分比
+ `dehaze_model(input_shape = INPUT_PATCH_SHAPE)`定义了模型，模型的结构及原理见论文。
+ `train()`定义了训练过程，使用了`model`的`.fit()`方法进行训练。其中定义了使用TensorBoard对训练过程进行可视化，以供训练时对训练过程进行分析，判断是否出现问题以提早停止训练，避免浪费时间；同时定义了callbacks的earlystop，在验证完提出的模型起作用后再通过earlystop实现更好的训练效果，在早期验证模型的过程中不用earlystop策略。可以在加入更多的callbacks的策略，另外可以重构该`train()`过程，通过逐步优化获得详细的训练过程数据。
+ 训练完的模型保存在`model_output`文件夹中，可以通过修改`demo.ipynb`加载的模型来验证训练的模型的有效性。

`demo.ipynb`：对单一图片进行去雾，通过修改其中的模型地址与待去雾图片地址，验证生成的模型去雾效果。

`dehaze.ipynb`：包含了大气光值估计、利用前面生成的模型对透射率分布图进行粗估计、导向滤波细化粗透射率分布图、通过大气散射模型复原无雾图像。同时包含了批量测试图片，在`demo.ipynb`中验证完训练的模型实现较好的去雾效果后，通过该文件对雾图进行批量处理，方便后续做定量分析对比。

`Evaluate.ipynb`：包含了图片进行PSNR、MSE、SSIM分析的代码。

+ 使用说明：
  1. 首先通过`Utils.ipynb`创建`evaluate`目录，然后将雾图及对应的真实无雾图片通过`Utils.ipynb`的相应部分加载到该目录下的`hazy`与`groundtruth`文件夹内。通过`dehaze.ipynb`对`hazy`文件夹内图像进行批量去雾处理，得到的去雾图片保存在`proposed`文件夹内。另外，将待对比算法的去雾图片保存至`evaluate`文件夹内相应的以算法命名的文件夹内。
  2. 在使用各算法实现去雾后，`evaluate`文件夹内的各文件夹都包含了相应算法的去雾图像，查看修改`Evaluate.ipynb`最后的`methods_list`，该列表中的各项为待对比的各算法及原始雾图与真实无雾图像进行PSNR,SSIM,MSE的定量分析对比。
  3. 完成第二步后会显示相应的信息，在`evaluate\result`文件夹中可以看到各种算法对于每幅图片的定量评价的数据，在`summary.txt`中保存了汇总信息。

`Utils.ipynb`：包含了很多工具（从数据集中提取指定数量的雾图及对应无雾图片到制定目录，创建进行定量测评的目录，使用MSD深度图创建雾图）：

+ `pick_hazy_and_gt_img(src_hazy_dir_path, src_gt_dir_path, dst_hazy_dir_path, dst_gt_dir_path, number, suffix, random_tag=1)`实现了从NYU2数据集中随机抽取一定数量的雾图到目标文件夹中，以进行批量图片去雾，并进行定量分析对比，其中：

  + `src_hazy_dir_path`与`src_gt_dir_path`：雾图与对应清晰图像的地址
  + `dst_hazy_dir_path`与 `dst_gt_dir_path`：目标地址

+ `dir_create(root_path, folder_list)`：在根目录下创建`evaluate`文件夹，在`./evaluate` 文件夹中 ：`groundtruth`、`hazy`、`proposed` 分别为清晰图片、通过大气散射模型生成的带雾图片、使用本算法去雾后的图片；其他算法得到的去雾图片的文件夹以算法英文简称命名，如DCP、MSCNN；result中存放了对比结果，`summary.txt`文件中有所有方法所有指标的总结，其他文件为具体图片结果细节。

  图片命名说明：以雾图'1381.jpg'为例说明：该图对应的真实无雾图片(ground truth)命名为'1381_gt.jpg'，用本算法去雾后图片命名为'1831_Dehaze_proposed.jpg'，以其他算法，如DCP算法去雾后图片命名为'1831_Dehaze_DCP.jpg'。

+ `haze_generator(msd_dir_path, output_dir_path, generator_num_per_img=5)`，利用MSD数据集的深度信息合成雾图，其中：
  
  + `generator_num_per_img`指定每幅图片生成的雾图数量。

#### 对比算法说明

本次毕设提出的去雾算法共与八种(ATM BCCR CAP DCP FVR DehazeNet MSCNN NLD)不同的去雾算法进行对比，其中DCP用Python实现，其他的其中是用matlab实现的。

在`compare_methods`中，包含了这八种算法的源码，每个算法的源码中都写了一个`batch_dehaze.m`或`batch_dehaze.py`文件，**在进行批图像去雾前，需要修改相关信息**：

+ 将`batch_dehaze.m`中的`hazy_path`与`dehazy_path`都修改为你电脑上的相应地址的绝对路径，注意要与代码中的格式相同，**并且在最后面要加`/`**
+ 将`batch_dehaze.m`中的`hazy_img_list`中后面文件的格式，如`'*.png'`修改为你要处理的图片格式，如果你要处理的图片的格式有多种，你可以改写这段代码。
+ DCP的`batch_dehaze.py`也要修改相关的路径信息。

++++++++

### 实验过程总结：

`train.ipynb`建模->`train.ipynb`训练->`demo.ipynb`,`dehaze.ipynb`验证模型是否有效，如果效果不佳，重新训练或者改进模型或者增强数据集，直到能实现较好的去雾效果->`Utils.ipynb`创建`evaluate`目录，并将`data`内数据集需要的图片加载到该目录下->通过`dehaze.ipynb`与`compare_methods`中各算法对图像进行去雾->通过`evaluate.ipynb`对各算法实现的去雾效果进行定量分析总结。















