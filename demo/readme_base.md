模型基础(baseline)

### 设计
**关于总体结构**
1. 骨架网络
    - vgg16: 主要采用2个3x3和3个3x3卷积作为模块进行堆叠
    - resnet18: 主要采用
    - resnet50: 主要采用
    - mobilenet: 主要采用1x1+3x3的模块进行堆叠，1x1负责调整通道数，3x3负责调整模型尺寸(下采样)
    ```
    
    ```
2. neck
    - FPN: 主要包括1组1x1, 一组上采样+融合，一组3x3，其中1x1为了把通道数调成一致，上采样则把尺寸调成一致，通道数和尺寸一致了才能进行相加融合。
      而最后的3x3则是为了消除叠加效应。
3. head
    - cls_head: 一般采用一个卷积调整通道数，配出21*n_anchor的通道数，目的是最后h*w*n_anchor正好是总的anchor数，而剩下的数字正好是分类数。
    - bbox_head: 一般采用一个卷积调整通道数，配出4*n_anchor的通道数，目的是最后h*w*n_anchor正好是总的anchor数，而剩下的数字正好是回归坐标数。
    - ldmk_head: 一般采用一个卷积调整通道数，配出10*n_anchor的通道数，目的是最后h*w*n_anchor正好是总的anchor数，而剩下的数字正好是回归坐标数。
    - head出来之后的标准操作：PRC(permute + reshape + concate)


**关于特征融合**
0. 核心：
    - 特征融合的目的是让浅层特征和深层特征在计算loss之前就进行融合，从而让浅层特征能够具有深层的语义信息，更有利于小物体检测。
1. FPN模式：把深层特征上采样放大尺寸，然后直接跟浅层特征数值相加。是从最深一层开始，逐层往浅层相加，让每一个浅层都能获得深层的语义，最浅层能获得前面所有层的语义信息。
2. SSH模式：

**关于如何定义训练target**
0. 核心：


**关于anchor尺寸和生成机制**
0. 核心：
    - anchor的目的：对比一个普通函数y=f(x), 神经网络本质上就是一个万能函数yi = f(x_img)，img经过神经网络，变换成多层特征图，然后映射成多个变量(y1,y2..yn)作为预测输出，也就是一个万能函数的特性
      普通函数fn=a1x^2+a2x+a3要得到fn函数的参数a1,a2,a3需要先输入一组x1,x2,x3, y1,y2,y3,然后拟合通过求解得到a1,a2,a3权重。
      同理对于神经网络，为了得到fn的权重，需要输入x为img, fn为神经网络，y1..yn为预测值(可以是score, distance..)，通过训练得到神经网络的权重。
      而anchor就是为了获得y1..yn的手段，只有得到y1..yn，才能对神经网络fn进行拟合，也就是训练，得到网络权重。
    - 神经网络的预测过程，本质上是对图片上多层固定方框集合的类别或坐标进行预测，anchor的目的则是获得万能函数输出y1,y2..yn的真值，来训练拟合函数求得函数的权重值。
    - 常用生成anchor的方式1：聚类法，通过聚类算法评估选出来的一组anchor跟整个数据集的所有anchor的mean_iou值的大小
    - 常用生成anchor的方式2：先确定anchor的最小尺寸，定义出base size, scale, ratio


**关于输入图片尺寸**
0. 核心：输入的一个batch图片训练前必然需要统一到同一尺寸，目的是能够进行堆叠，从而高效训练。而尺寸调整需要考虑：
    - 如果是小目标检测，比如人脸，尽可能不要采用缩减变换，比如直接rescale会导致小目标变得更小，不利于训练。
    - 如果目标形状比较固定，比如方形人脸，尽可能不要改变缩放比例，防止训练目标的形状比例跟anchor尺寸不匹配。

**关于样本不平衡**
0. 核心：
    - 样本不平衡的影响：如果正负样本不平衡，导致的问题是负样本的总的损失之和远超正样本，从而反向传播的梯度被负样本的影响主导，正样本被淹没。
    - 改变样本不平衡的方法1：对样本采样，提取少量负样本，让正负样本的比例控制在1:3 (之所以是1:3因为正样本的损失值一般比负样本大很多，用1:3相当于平衡一下取值)
    - 改变样本不平衡的方法2：对损失采样，提取少量负样本的损失，其他损失不计入总损失，从而防止反传梯度被负样本控制
    - 改变样本不平衡的方法3：对损失加权，

**关于损失函数**
0. 核心：
    - 分类损失
    - 回归损失


**关于损失函数的权重取值问题**
0. 核心：


### 调试

**关于下采样和上采样**
0. 核心：




**关于设置获取target时正负样本的分割阈值**
0. 核心：


**关于设置负样本挖掘时正负样本比例的问题**
0. 核心：


**关于数据格式**

0. 默认日常描述图片尺寸，采用[w,h]的形式，比如一张图片是1280*800就是指宽w=1280, 高h=800。
   因此在cfg中所指定img scale = [1333, 800]就是指w=1333, h=800
   从而转入计算机后，要从w,h变成h,w
1. 默认的大部分数据集，输出格式都是n,h,w,c和bgr格式，一方面是hwc更普遍，另一方面是opencv读取的就是bgr。
2. pytorch中指定的数据格式是chw和rgb(非常重要！记住！)，所以常规处理方法是：数据集输出都统一定义成hwc和bgr，再通过
   transform来转换成chw和rgb
   
   
**关于img/label与模型weight之间的数据格式匹配**

1. 输入img要修改为float()格式float32，否则跟weight不匹配报错
   输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错
   img = img.float()
   label = label.long()
   这两句要放在每个batch开始位置

   为了避免遗忘，可以把这部分操作集成到自定义的to_tensor()函数中，在每次开始转tensor的时候自动转换。
   但对于numpy形式的img, label, segment这里无法区分，需要在数据集中单独转换为float32或int64
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
        
2. 如果要把在GPU中运算的数据可视化，必须先变换到cpu，然后解除grad，最后转numpy才能使用。
   即：x1 = x.cpu().detach().numpy()
   
3. 同样，只有都在同一设备上的数据才能进行相互运算，比如GPU中的数据可以相互运算，但是GPU中的数据不能跟cpu中的数据进行计算。
   所以在pytorch中需要比较小心，尤其是创建新数据时，务必要把device带上。
   而pytorch为了方便，提供了一系列新建tensor的命令，能够直接从其他tensor中继承device
   '''
   a = torch.tensor([1,2,3], device=torch.device('cuda'))
   b1 = a.new_full()   # 从a继承device, 可再定义size, dtype
   b2 = a.new_zeros()
   b3 = a.new_ones()
   '''


**关于图片标签值的定义在分类问题和检测问题上的区别**

1. 在纯分类任务中，数据集的label一般定义成从0开始，比如10类就是[0,9]，这样的好处是在转独热编码的时候比较容易，
   比如标签2的独热编码就是[0,0,1,0], 标签0的独热编码就是[1,0,0,0]
2. 而在物体检测任务中的分类子任务中，一般会把数据集的label定义成从1开始，比如10类就是[1,10], 这样做的目的是
   因为在检测任务中需要对anchor的身份进行指定，而比较简洁的处理是把负样本的anchor设定为label=0。所以相当于把
   label=0预留给anchor的负样本。


**关于transform中涉及的类型变换导致的错误**

1. transform和transform_inv中涉及的数据类型变换很多种类，很容易漏掉没有做而导致输出形式不对。

2. 对于transform_inv的变换，需要重点关注
    - 默认数据集输出类型：hwc, bgr。采用这种默认输出形式，主要是因为用opencv作为底层函数的输出就是这种形式。
      而pytorch需要的形式是chw, rgb，所以经过transform后输出就是chw,rgb
    - 逆变换需要先变换chw为hwc，然后才变换rgb为bgr：因为rgb2bgr是基于最后一个维度是c来写的。
    - 逆变换需要把



**关于训练过程中图片尺寸如何变换的问题**

1. 通常会定义一个边框尺寸，比如scale = (300, 300)，这是图片的最大尺寸范围。

2. 图片首先经过transform，按比例缩放到边框尺寸，此时因为比例固定，所以每张图片尺寸都不同，但都有一条片跟边框尺寸拉平相等。
   比如一个batch的图片可能尺寸会变成(300, 256),(300, 284),(240,300)这样的形式。

3. 图片然后经过dataloader的collate_fn，对一个batch的图片取最大外沿，进行padding变成相同尺寸的图片。
   由于transform时所有图片已有一条边靠近边框尺寸，所以取所有图片最大外沿结果基本都是边框尺寸，比如一个batch的图片会变成
   (300,300),(300,300),(300,300)然后堆叠成(3,300,300), 所以最终进行训练的图片基本都是以边框尺寸进行训练。


**关于训练时候的路径问题**

1. 经常会产生路径找不到的情况，比较合理的解决方案是：
   - 数据集的root路径，尽可能采用绝对路径，即以/开头的绝对路径。
   - 项目的root路径尽可能加到sys.path中去。

2. 如果有报错说部分路径无法导入，一般有2种可能性：
   - 根目录路径不在sys.path中，可通过添加根目录路径到sys.path并结合根目录下一级目录可见的原则，实现大部分文件的导入。 
   - 被导入的module中含有错误的导入，需要修正这些错误，才能解决报错
   - 存在交叉导入：即A要导入B，同时B也要导入A，此时有可能产生错误，需要解除交叉导入才能解决报错。
   
3. 一句话添加系统路径：sys.path.insert(0, os.path.abspath('..'))


**关于归一化和标准化**

1. 常见训练采用的mean, std参数(均来自mmdetection)：
   这些训练所采用的mean,std并不是跟训练数据集相关，而是跟基模型所训练的数据集相关，这是
   因为这些训练都是在基模型的基础上进行finetunning做迁移学习来训练的。
   由于pytorch的基模型基本都是在imagenet中训练的，所以这些mean, std都是imagenet的参数。
   而caffe的基模型虽然也是在imagenet中训练的，但因为处理方式不同所以std取成了1(待在caffe中确认原因)
   比如：
   - 来自pytorch的基模型：[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
   - 来自caffe的基模型：[123.675, 116.28, 103.53], std=[1, 1, 1]

2. 如果在一个数据集上从头训练，则需要事先计算他的mean, std。但要注意mean,std的数值顺序到底是BGR顺序还是RGB顺序。
   如下是参考的数据集数据(来自mmdetection):
    - cifar10(归一化加标准化): mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]
    - cifar10(只做标准化)：mean, std

3. pytorch自己的transform往往是先归一化再标准化，即：
   img = (img/255 - mean) / std
   所以事先准备的mean和std都是很小的值，在-1,1之间

4. 后边看到一些其他例子，比如mmdetection里边对其他数据的提供的mean, std值较大。
   说明对数据的处理是只做了标准化，即：
   img = (img - mean)/std
   所以实现准备的mean和std都是比较大的数值，在0-255之间    
   
5. 总结一下就是：
    - 如果是用现有模型参数做迁移学习：采用的mean/std需要跟原来基模型一样，数据预处理也要按这种方式。
      比如采用caffe/pytorch基模型，则都是只做标准化是img = (img - mena) / std
    - 如果从头开始训练：则采用的mean/std跟实际img的预处理方式必须一样。
      比如采用在0-1之间的小的mean, std，则实际数据处理也要要归一化+标准化。
      而采用大的mean,std，则实际数据处理也只要做标准化。


**关于卷积的取整方式**

1. pytorch中conv, maxpool在计算输出w,h尺寸时，默认的取整方式都是下取整(floor)。
   唯一不同的是，maxpool可以手动设置成上取整即ceil mode，但conv不能手动设置，也就只能下取整。

2. conv, maxpool两者计算输出尺寸的公式一样
    - 没有dilation时，w' = (w - ksize + 2p)/s + 1
    - 有diliation时，相当于ksize被扩大，此时
                     w' = (w - d(ksize-1) +2p -1)/s  + 1
                     
                          
      
**关于神经网络的前向计算和反向传播在pytorch中的对应**

1. 前向传播：就是模型一层一层计算，output = model(img)
2. 损失计算：采用pytorch自带的nn.CrossEntropyLoss(y_pred, y_label)，则不需要手动增加softmax，
   也不需要手动把label转换成独热编码。因为这两部分都被写在pytorch自带的交叉熵函数对象内部了。
3. 损失反向传播：必须针对损失的标量进行，也就是losses先做规约缩减(reduction='mean')，然后才能
   loss.backward()，即这里loss是一个标量值，pytorch对这个标量内部梯度会自动获取，并反向传播。
4. 优化器更新权值：必须采用optimizer.step()完成
5. 额外要求：必须增加一句优化器梯度清零，optimizer.zero_grad()，这句必须放在backward之前，
   确保变量的梯度不会一直累加，而是每个batch独立计算，一个batch结束就清零一次。
   (自己写的神经网络，梯度是整个batch一起算，不需要累加，计算以后直接赋值，所以也就不需要清零了。) 



**关于在GPU训练**

1. 如果要在GPU训练，只需要3步
    - 创建设备device:  device = torch.
    - 模型送入device
    - batch data送入device: 注意这里其实只要img送入device就可以，因为跟model相关的计算只需要img输入

2. 并行式GPU训练并不一定比单GPU快，相反对于一些比较小的模型，单GPU的速度远超过并行式训练的速度。
   可能因为并行式训练需要让数据在GPU之间搬运造成时间损耗，同时python的并行式训练并不是真正的并行，
   而是在同一时间下只有一块GPU运行的假并行，只是能利用多GPU内存而不能利用多GPU算力的假并行。

3. 分布式训练才是真正的多GPU算力并行训练。




**关于如何设置DataLoader**

1. 对常规的数据集，如果图片尺寸都是一样的，那么直接使用pytorch默认DataLoader就可以。
2. 对数据集中img的尺寸不一样的，由于dataloader需要对img进行堆叠，此时图片尺寸不同无法直接堆叠，
   则需要自定义collate_fn对img进行堆叠，同时处理那些labels/bboxes/segments是堆叠还是放入list.
   
3. pytorch默认的collate_fn设置不是None而是default_collate_fn，所以即使不用collate_fn
   选项，也不要去把collate_fn设置为None，这会导致collate_fn找不到可用的函数导致错误。
   (从这个角度说，pytorch的官方英文文档有问题，注明DataLoader的collate_fn默认=None，
   但实际上collate_fn的默认=default_collate_fn。)

4. 关于batch size: 如果是单机并行训练，则batch size代表的是所有GPU加载的图片总数
   而如果是多机分布式训练，则batch size代表的是单个进程也就是单块GPU加载的图片数
   

**关于contiguous的问题**

1. numpy和pytorch都有可能产生in_contiguous的问题。主要都是在transpose(),permute()之后会发生。
参考：https://discuss.pytorch.org/t/negative-strides-of-numpy-array-with-torch-dataloader/28769

2. 在numpy数据下的报错形式是：ValueError: some of the strides of a given numpy array are negative. 
   This is currently not supported, but will be added in future releases.
   
3. 在tensor数据下的报错形式是：
   可通过t1.is_contiguous()查看到是否连续。

4. 解决方案;
    - 对于numpy：  img = np.ascontiguousarray(img)
    - 对于tensor： img = img.contiguous()


**关于在dataset的__getitem__()中增加断点导致程序崩溃的问题**

1. 现象：如果模型和数据送入GPU，dataloader会调用dataset的__getitem__函数获取数据进行堆叠，
   此时如果在__getitem__里边有自定义断点，会造成系统警告且暂停训练。
2. 解决方案：取消断点后模型训练/验证就正常了。
    - 如果想要调试__getitem__里边的语法，可以设置额外的语句img, label = dataset[0]来进入__getitem__进行调试。
    - 如果想要调试collate_fn里边的语法，可以设置workers=0, 然后在collate_fn添加断点，再通过next(iter(dataloader))进行调试。



**关于预训练模型的加载**

1. 预训练模型加载需要分两步：第一步加载checkpoint文件，第二部加载state_dict，且第二步需要去除多余的key以及去除param size不匹配的key.
2. 对于state_dict不匹配的问题，一般两种解决方案：
    - 直接修改预训练模型的最后一部分，然后加载参数时过滤不匹配的key
    - 自己创建整个模型，然后加载参数时过滤不匹配的key
3. 预训练模型的参数如果放在默认.torch/model或者.cache/里边，则作为隐藏文件夹中的文件，无法通过os.path.isfile()检查出来，会报错找不到该文件。
   所以，虽然能够加载但无法检出。
   也可以把参数移到自定义文件夹去，就可以通过os.path.isfile()检测到了。
   

### 关于常规分类问题和物体检测中的分类问题的差异？

1. 常规分类问题是对一张图片作为最小个体进行分类；而物体检测问题中的分类是以一张图片中的每一个bbox进行分类。
   因此对于物体检测问题本质上一张图片是多个分类问题的集合。
   
2. 因此常规分类问题是一张img对应一个独立label, 1个batch的数据为多张img对应多个独立label，1个batch计算一次loss，
   所以需要把一个batch的label组合成一组，相当于1个batch就是一组img对应一组label。
   因此分类的一个batch计算本质上相当于检测问题的一张图片计算。

3. 但是检测分类问题是一张img有一组bbox对应一组label，1个batch的数据为多张img拥有多组bbox对应多组label，每组bbox,label完成一次loss计算。
   因此检测的一个batch计算本质上相当于每张图片是一次分类batch，多张图片就要手动进行多次img循环计算，循环的主体就是常规分类问题的一个batch。
   这样理解的话，就可以统一常规分类问题和检测中的分类问题了：
       - 常规分类问题需要把labels组合成一组，变成一个标准计算。
       - 检测分类问题需要循环计算每张img，而每张img计算相当于一次常规分类问题的batch计算。


### 关于卷积核的作用

1. 结论：
    - 浅层卷积核主要过滤边缘、纹理的初级信息；
    - 中层卷积核就开始学习小物体；
    - 深层卷积核开始学习大物体；

2. 证明：参考https://www.leiphone.com/news/201808/DB6WARlVGdm4cqgk.html
   可以看到单层(也就相当于浅层)卷积核收敛以后的输出图片，类似于sobel算子，是对图片进行某个方向的边缘过滤。
   也就是说明了浅层神经网络主要是学习边缘、纹理等初级信息


### 关于新的一些卷积层结构
0. 小卷积代替大卷积：对于滤波器来说，并不希望滤波核太小，这样能够过滤空间太小(也就是感受野太小)，主流滤波器尺寸是5x5, 7x7。
   然而大滤波器缺点是参数太多(out_c, in_c, h,w)。
   而采用小卷积串联的方式，就能够实现较大感受野的滤波器效果。
   同时小卷积串联的方式，参数量相对于大卷积核来说要更少，比如2个3x3参数量是2x3x3=18, 而1个5x5=25。
   同时小卷积串联的方式，引入更多非线性激活模块，有利于模型的表达能力提高。
   
    - 两个3x3卷积核，就相当于一个5x5卷积核的感受野(5)
    - 三个3x3卷积核，就相当于一个7x7卷积核的感受野(7)
    - 五个3x3卷积核，就相当于一个11x11卷积核的感受野(11)，比如alexnet第一个卷积是11x11的，可以用5个3x3代替

1. 最新的1x1小卷积，产生的新的作用：
    - 能实现卷积基本的调整特征通道数，即实现降维或升维。
    - 配合3x3来先降维再升维：比如1x1+3x3+1x1，相当于先用1x1降维，然后3x3学习，然后1x1再升维，从而可保证整个模块维度不变，便于跳线叠加(resnet)

2. 分组卷积：可以非常有效的减少卷积参数。(参考resnext的实现)

   
1. 案例1：vgg中的3x3卷积，采用2到3个3x3卷积核的串联，实现感受野为5到7的滤波器效果(滤波器大小为5或7应该是计算机视觉中主流的滤波器形式)

2. 案例2：resnet中的1x1卷积，可用来实现卷积层基本功能改变通道数，并且主要用来减少参数
    - resnet中的1x1+3x3+1x1结构：1x1用来先减少通道数(从而减少参数)，然后让3x3做过滤，然后再用1x1恢复通道数


**关于全连接层的参数过多如何解决的问题**

1. 全连接层所占参数一般占了一个模型总参数个数的80%以上，所以为了减小模型所占内存，需要替换全连接

2. 对比：
- 方式1： 全连接
    ```
    x = self.feature(x)
    x = self.avgpool(x)            # (b,c,h,w)
    x = x.reshape(x.shape[0], -1)  # (b, -1)
    x = self.fc(x)                 # (b, 20)
    ``` 
- 方式2： adaptiveAvgpooling
    ```
    
    ```
- 方式3：卷积
    ```
    x = self.feature(x)                     # (b,c,h,w)
    x = self.conv1x1(x)                     # (b,c',h,w), 其中c'是为了配出需要的类别数
    x = x.permute(0,2,3,1).reshape(-1, 21)  # (b, 21)
    ```


替换方式1：resnet/resnext的最后一部分就是这样实现。
   用一个adaptiveAvgpooling(1), 先把卷积的每个通道收缩为1，即(c,h,w)->(c,1,1)，再reshape后接一个简单全连接linear(c, 10)即可
   变换前reshape后需要至少2个全连接(c*h*w->256),再(256->10)，参数总量256*c*h*w + 256*10；
   avgpool变换后reshape后只需要1个全连接(256->10)，参数总量为256*10, 减少了256*c*h*w个参数。
   
   
   


### 关于GPU训练完成后所占用的GPU内存无法释放的问题

1. 通过查看nvidia-smi发现训练完成后GPU的内存依然被占用没有释放，在spyder中clear或者torch.cuda.empty_cache()都没有用。
   除非关闭当前训练终端重开一个终端。



### 关于如何提升精度的tricks

1. 没有BN可以训练，但增加卷积之后激活之前的batchnorm2d()层，以及全连接之后激活之前的batchnorm1d()层，
   可至少增加10%以上的精度。
   同时：如果是从头训练，建议用GN即组归一化代替BN，因为GN对batch size不敏感，具有比BN更稳定的性能。
   但如果是采用已有预训练参数，则不太方便把BN改成GN，除非在训练的时候也让GN不更新参数，比如重写
   带GN的model的train()方法，默认train()方法是把BN/Dropout模型设置为训练模式，重写train()方法就可以手动
   关闭model中GN的更新，可参考mmdetection中关于resnet的train()方法。
   
2. 没有初始化而采用pytorch默认初始化值可以训练，但增加恰当的初始化手段(xavier_init/ kaiming_init)，可提高精度

3. 固定学习率可以训练，但增加warmup的可变学习率，可提高精度

4. 训练精度如果已足够高，比如达到1或0.99，但验证精度不高，比如只有0.8，此时说明训练过拟合了。



### 关于mmdetection是如何把数据打包，以及如何送入设备端的？

0. 先写结论：通过重写data_parallel_model以及里边的scatter()函数，来拆开data_container，并送model, batch_data到指定gpu设备。
   并通过to_tensor函数，把数据转换成pytorch需要的float(), Long()，防止了训练出错。
   整个过程封装得非常隐蔽，虽然减少了用户出错的几率，但也让使用者不清楚应该有什么是需要做的，有什么是系统帮忙做掉的，在哪做掉的。
   
   具体来说：
    - 定义了一个data container，用来打包数据
    
    - 定义了一个MMDataParallelModel对模型进行打包，这个包继承自DataParallel但重写了scatter函数。
       目的是在scatter函数中对data container拆包，并对data container里边的数据送入device.
       因此在mmdetection中无需增加data.to()这步手动操作，同时也无需手动去除data container的外壳，
       而是在scatter函数中自定义处理data container的过程，以及data送入device的过程。
    
    - 自定义了collate_fn



### 关于pytorch中数据筛选小技巧

1. 筛选值：
    - np.max() -> torch.max()
    - np.min() -> torch.min()

2. 筛选序号：
    - np.where(data > k) -> torch.nonzeros(data > k)  # 为了获得筛选序号，numpy中有np.where()这种大杀器，在pytorch中则需要使用torch.nonzeros(data > k)这种组合方式达到相同效果
    - np.argmax()        -> torch.argmax()    
    - none               -> values, inds = torch.max(), 
    - none               -> values, inds = torch.min()  # pytorch中对max,min进行了增强，不仅可以获得筛选值还能获得筛选序号，values, inds = torch.max(data, dim=0)
    - none               -> values, inds = torch.topk()


### 关于如何快速训练和验证结果

1. 完整跑一个coco，甚至是一个voc去训练，时间是很久的。毕竟一个voc(07+12)就有15,000张图片，单GPU一个epoch跑下来就要5-6个小时。

2. 有时为了调试需要，或者为了快速看一下结果，可以采取如下方式加快训练速度：
    - 只用voc07：5000张图片，数据集少了2/3
    - 只跑1-2个epoch：每个epoch不要叠加数据集，这样1个epoch也就5000张图片。


### 关于在opencv中绘制rectangle却总是显示不出来

1. 原因在于自己初始化的一张图片img是(300, 300)，也就是一张灰度图，且被自己设置成黑色背景了，
   导致所有线条只要不是白色，就是黑色，完全淹没在背景中了。

2. 另一个原因是：opencv显示的图片数据格式必须设置为uint8，否则显示不正确，这个问题也搞了好久。



### 其他一些犯过的错误

1. 一般会记得把img转换成float(), label转换成long(), 但往往忘了把weight权重转换成float()，而他会跟输出的loss相乘，所以一定要为float()
2. 图片在transform做normlize时，通常提供一个mean,std，但往往忘了关注mean里边值的顺序是RGB还是BGR，需要注意这个顺序跟normlize时的img形式相关。
   如果img是先做bgr2rgb再做normalize，则提供的mean必须是RGB顺序；而如果img是先做normalize再做bgr2rgb则提供的mean必须是BGR顺序。
   通常来说，拿到的mean一般是RGB的，在自己的算法框架里边，一般也是先做bgr2rgb，所以要拿到的mean都用RGB形式。

3. 第一次跑自己的ssd，直接就爆出cuda错误，单步查看loss发现发现损失爆炸了，loss太大可能的原因是我代码写错了，或者设置错了，
   查了下整个模型的输出，发现每层的输出很正常，在没有bn的情况下一开始输出算合理，但到后边就慢慢变高，我突然想是不是lr太大，
   查了下发现自己的lr设置成了0.01, 缺失有点大，于是改成0.001，就可以正常训练了。

    合理的调试过程是：
    
    - 逐层查看激活输出的均值/方差：也可以绘制出每个激活层输出的分布图，没有bn的激活输出一般变化比较大，但也应该在+-几百之内，并且不能继续增加。
      确保没有错误的层设置，导致激活值异常。如果均值方差偏移导致问题，那估计得加bn了。但如果初始化没问题加上有预训练权重，一般也能模型训练收敛。
    
    - 逐iter查看分类损失和回归损失：一般分类损失较大(单图2~10之间，batch之和在20左右)，回归损失较小(单图1-2之间，batch之和在5左右)，总的loss在30以下。
      确保没有突然的激活变化。
      
    - 一个重要调试方法：哪里报错就在哪里设置print()打印可能的size,shape或者值，看看这种报错到底是什么值造成的，然后再取修正这种情况。

4. 做预测也就是取获得get_bbox()的时候，输出的预测格式非常重要且容易出错。正常的预测输出单图预测一般是bbox_pred(k, 5), label_pred(k,)，
   其中bbox的5列包含4列坐标和1列置信度。但是这种输出并不便于进行mAP的计算，所以需要转换成另一种单图预测result(n_class, )(k, 5)，其中
   n_class表示按类别获得预测bbox，每个类别都有输出一组预测bbox(k, 5)，且由于是按标签顺序输出的，所以标签也就不需要输出了。
   多张图则为(n_imgs,)(n_class,)(k, 5)的形式，也就是4个维度(n, c, k, 5)，只不过用list嵌套list嵌套array的形式表示了。
   这种数据格式提供给mAP函数接口，就可以就算模型针对某个验证数据集的mAP了。

5. 报错了一个TypeError: 'torch.Size' object is not callable，我把那段程序从头到位看了20遍，一个字母一个字母对过去，也没发现问题在哪。
然后找了个类似的不报错的程序，又一句话一句话复制粘贴替换，最后终于找到问题，我把data.shape[0]写成了data.shape(0)，这个错误浪费了我2个小时。


### 训练进行时突然报错，训练无法进行，验证也无法进行，然后定位到dataloader, 报错内容是OSError:[Errno 12] Cannot allocate memory

1. 这种错误表示无法分配存储空间，可能的原因有：
    - 显存不够：此时可以检查自己的batch size是不是设置过大，导致显存不够，通过nvidia-smi查看显存占用情况
    - 需要加载的数据较大，而设置的存储空间不够，但workers又较高，此时可以减小workers，比如workers=0
    - 硬盘空间不够：也有可能导致，需要释放一些硬盘空间。

2. 我第一次碰到是因为前一次的batch size 设置过大，所以把batch size改小即可，但需要重新启动一个kernel
   我第二次碰到是因为前一次运行别的比较大的数据集(比如coco)，然后再跑小的数据集也会出现该问题，此时把workers设置为0才可以，重启kernel都没有效果
   参考：https://discuss.pytorch.org/t/oserror-errno-12-cannot-allocate-memory/24827/3


### 训练中报错，gt_bboxes无法切片

1. 这个问题之所以单独拿出来说，因为gt_bboxes的处理很容易在数据集处理时遗漏对特殊情况的处理，比如bboxes为空，这种情况会在模型中多处产生报错。
所以一定要在数据集代码区把这个问题处理干净。好的办法是在__getitem__()方法中用while True来获取idx数据，如果len(bbox)=0则随机一个新的idx并continue，
直到得到bbox正常的数据才return



### 训练中报错variable has been modified by an inplace operation：

1. 详细报错信息如下：似乎leaky_relu输出设置成inplace方式不对。
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: 
[torch.cuda.FloatTensor [8, 64, 40, 40]], which is output 0 of LeakyReluBackward1, is at version 2; expected version 1 instead. 
Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

2. 该报错非常少见，后来在pytorch官网论坛上找到2个帖子，说明了这个问题产生原因：
https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
https://discuss.pytorch.org/t/defining-my-model-encounter-a-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/27266

原因分析：
主要是在代码定义时，把某些需要反向传播的变量进行了修改，比如x += 1这是一种in place操作，而x = x + 1则不是in place操作，
而是先操作再赋值。对于in place的操作如果该变量同时又被别的操作所引用产生修改，就会打乱反向传播，造成错误。

所以我反复寻找代码中类似x +=的操作，最后在FPNSSH中找到一处，fpn_outs[i-1] += F.interpolate(fpn_outs[i], scale_factor=1, mode='nearest')
改成fpn_outs[i-1] = fpn_outs[i-1] + F.interpolate(fpn_outs[i], scale_factor=2, mode='nearest')即可



### 验证输出的bbox的尺寸跟实际有较大偏差，主要原因是bbox转换出错？
1. bbox的转换过程：
    - img/bbox预处理resize
    - img/bbox显示时可以直接按照img_raw和bbox来显示即可
    - img/bbox变换分辨率。
2. 特殊情况：yolov3，因为他的bbox输出时是相对于处理之后的img来说的，也就是相对尺寸。
    - img/bbox预处理resize(608,608)
    - img/bbox显示时采用原图img_raw，bbox则需要用原图尺寸相乘，类似于如下操作，则可得到基于img_raw的bbox实际尺寸。
      bbox采用相对尺寸的好处是，无论要基于什么尺寸的图片进行显示，直接乘以图片尺寸就可以了，而不需要其他操作。
```
width, height = img_metas[0]['ori_shape'][1], img_metas[0]['ori_shape'][0]   # 改成用进入模型的图片尺寸来恢复bbox，而不是用output_resolution
image_dims = [width, height, width, height]
boxes = boxes * image_dims
```
    
    
### 关于如何提高小物体的检测精度

1. 小物体的检测精度不高，主要有2个原因，第一个是模型没有学习到，所以误检；第二个是模型学习到了但没有检测到，所以漏检

2. 对于模型没有学到产生的误检：
   解决方法之一是让模型学到以后再输出，比如ssd检测算法，采用的小物体检测特征图是从第10层输出的，这层比较浅，所以模型还没有学习到，导致的是误检。
   解决的办法是增加模型深度，让小物体检测特征图从相对更深的层输出，比如改成resnext101，从第??层输出就会更好一些。
   
   另一种解决办法是增加特征的融合，让浅层(位置信息准确但语义信息少)与深层(位置信息粗略但语义信息丰富)的特征进行叠加融合，
   从而提高浅层特征的语义，提高小物体检测，比如采用FPN模块，自顶向下进行特征融合。

3. 对于模型学到了但没有检测到产生的漏检：
   解决方法1是调整anchor尺寸，让anchor能够匹配小物体尺寸。
   解决方法2是采用anchor free的方式，避免小物体受检测anchor的影响，比如采用FCOS算法中的中心点检测。
   



### 关于分割数据集的使用

1. voc的分割数据集：seg文件是一个png图片文件，里边通过PIL.Image加载以后就能直接得到取值0-20共计21个类别的像素值，还包括取值=255的白色边框线。
所以需要进行的处理就是，把255的值置0作为背景，然后把Image对象转换为numpy作为seg_label进行训练即可。
(参考：)

   
### SSD物体检测算法的总结

1. anchor系统的基本思想是什么？
    - 如何把检测问题转化为分类回归问题？
      转化的本质，就是要把整张图片特征转化成有位置特性的分组特征，这种分组可以看成特征的样本化。
      样本1： x,y,c,r,g,b 这些就是一个最简样本具有的特征，通过把特征样本化为这样的8000多个样本去训练。
      通过特征去训练，就相当于用样本训练，从而训练完了如果单个样本输入就能预测该样本的类型(label)和尺寸(xmin,ymin,xmax,ymax)
      样本不能只是gt_bbox，因为这样的样本太少，也缺少位置信息。
      
      anchor机制，就是一种特征样本化为anchor(映射为8000多个样本)，并且获得每个样本标签的方法：
      通过创建对任何图形都不变的anchor系统把特征转化成8000多个样本，同时通过iou获得这8000多个样本的真值标签，就相当于把检测问题转换成8000多个样本的分类问题。
      8000多个样本的特征被神经网络(也就是万能函数)学习，输出(8000, 21)就是预测结果，再跟anchor系统得到的标签(8000, 21)进行损失计算就可以让网络进行学习了。
      所以说只要有样本，有标签，就能够训练神经网络。
      
      而后续出现的anchor free的方法有另外一种获得各个子样本和对应标签的方法
   
2. ssd需要让所有输入图片尺寸转换成统一一致的300*300或512*512，主要原因在于他最后一层的缩减尺寸正好控制在1x1，如果图片不是严格控制
   尺寸的化，有可能导致卷积计算的输入尺寸小于卷积核本身从而报错。
   
   一个常识：基本上所有的基于anchor的算法或者基于anchor free的算法，由于都要对img进行网格划分(部署anchor或者部署关键点)，所以在img的输入尺寸上都有要求，不能是随便的一个尺寸。
   - 方式1：固定输入img的尺寸，比如ssd，就把输入图片固定为300*300，图片会有一定以尺寸比例差异。
   - 方式2：固定一个尺寸最大范围，然后通过padding的方式让图片的宽高成为某个数字的倍数，比如32的倍数，这样能够保证在划分网格的时候不会产生除不尽的情况。
     比如retinanet， fcos。

3. ssd的缺点和改进
   参考：https://blog.csdn.net/u010725283/article/details/79115477/
    - ssd的缺点：就是对小目标的检测效果不够好，根本原因是浅层特征的语义信息太少，产生误检。
    - 改进实例DSSD
        (1). 采用更深的网络resnet101来代替vgg16，这样浅层的特征来自于更深的网络，相应就有更多语义信息
                    out[0]          out[1]          out[2]          out[3]          out[4]          out[5]
            ssd     depth=13        depth=20        depth=22        depth=24        depth=26        depth=27
                    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)
                    
            dssd    depth=23        depth=101       depth=104        depth=24        depth=26        depth=27
                    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)    size=(38,38)
        (2). 采用？？？卷积来进行特征融合，让浅层特征也能包含更多的语义信息。
        
### YOLOV3物体检测算法的总结
1. backbone: 为darknet53也就是53个卷积层，总体结构如下(https://blog.csdn.net/qq_37541097/article/details/81214953)
```
img            3   416x416
conv  3x3      32

conv  3x3  s2  64  208x208     # 用一个conv3x3一方面下采样减小尺寸，一方面增加通道数 
1xconvs        64              # convs代表基础模块(conv1x1 + conv3x3)
residual                       # 用一个短路回路，保证模块级别的梯度不会爆炸或者消失

conv  3x3  s2  128 104x104     
2xconvs        128
residual

conv  3x3  s2  256 52x52
8xconvs        256
residual                       # (out2: 256x52x52) 

conv  3x3  s2  512 26x26
8xconvs        512 
residual                       # (out1: 512x26x26) 

conv  3x3  s2  1024 13x13
4xconvs        1024         
residual                       # (out0: 1024x13x13)

``` 
- backbone的基础模块是1x1+3x3的结构：每个这样的基础模块都会包含一个residual的短路回路(类似resnet).
跟resnet的区别在于：resnet的下采样是放在module里边来做的，也就是每组module的第一个模块会有下采样，所有对应residual分支上1x1也会带下采样。
而darknet的改变在于他单独出来一个conv3x3做下采样，这样所有的moduel就都统一了，residual的1x1也就不分是否带不带下采样，结构上比resnet更简洁。
```
Conv2d(64, 32, 1, stride=1, bias=False)             # conv1x1
BatchNorm2d(32)
LeakyReLU(negtive_slop=0.1)
Conv2d(32, 64, 3, stride=1, padding=1, bias=False)  # conv3x3
BatchNorm2d(32)
LeakyReLU(negtive_slop=0.1)

residual_layer                    # 采用add
```

- backbone输出的特征层的融合：
因此第一层输出是最深层输出，没有融合，第二层输出中中间层，跟最深的第一层进行了融合，融合方式是concatenate
第三层作为最浅层再跟前两层的融合结果做融合，融合方式concatenate。融合之前为了能够concatenate，都对前一层进行上采样，让两层特征size相同才能融合
注意：这里跟retinanet的区别在于，yolov3的特征融合是采用concatenate方式，而FPN的特征融合是采用add
```
# retinanet的FPN特征融合
conv1x1      # (64, 40, 40)
upsample     # (64, 80, 80)
add          # (64, 80, 80)+(64, 80, 80) -> (64, 80, 80)

# yolov3的特征融合
conv1x1      # (256, 13, 13)
upsample     # (256, 26, 26)
concat       # cat[(256, 26, 26),(512, 26,26)] -> (768, 26, 26)
```

- 最终的特征输出：特征融合之后就进入head模型(也就是这里的yolo_layer)，主要进行展平特征，生成anchor,计算损失
其中展平特征主要是让通道数匹配最后的需求(配出总计anchor个数)：比如85 = 80(每类置信度) + 4(坐标) + 1(类别)，768=16*16*3, 3072=32*32*3, 12288=64*64*3，也就是特征图尺寸*3个anchor 
```
// 比较好理解的输出
yolov3_outputs   #(3,) (1, 3*85, 13, 13),(1, 3*85, 26, 26),(1, 3*85, 52, 52)

// 该版本yolov3的输出(本repo)
yolov3_outputs    #(3,) (8, 768, 85),(8, 3072, 85),(8, 12288, 85)   # 注意这里768=16*16*3

// 其他版本yolov3的输出(onnx的输出)
yolov3_outputs    #(3,) (1, 255, 13, 13),(1, 255, 26, 26),(1, 255, 52, 52),  注意这里255=85*3
```

### 数据流汇总
- batch数据：img(8,3,480,480), targets(52, 6), 其中的6列中：第1列是图片号(0-7), 第2-5列是bbox相对图片的坐标(x,y,w,h), 第6列是标签
- backbone输出：(3,)(b, 256, 52, 52),(b, 512, 26, 26),(1024, 13, 13)
- yolo head输出：(3,)(b, 255, 13, 13),(b, 255, 26, 26),(b, 255, 52, 52)