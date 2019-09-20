
# 临时记录pytorch训练深度神经网络的注意事项


### 关于数据格式

0. 默认日常描述图片尺寸，采用[w,h]的形式，比如一张图片是1280*800就是指宽w=1280, 高h=800。
   因此在cfg中所指定img scale = [1333, 800]就是指w=1333, h=800
   从而转入计算机后，要从w,h变成h,w
1. 默认的大部分数据集，输出格式都是n,h,w,c和bgr格式，一方面是hwc更普遍，另一方面是opencv读取的就是bgr。
2. pytorch中指定的数据格式是chw和rgb(非常重要！记住！)，所以常规处理方法是：数据集输出都统一定义成hwc和bgr，再通过
   transform来转换成chw和rgb
   
   
### 关于img/label与模型weight之间的数据格式匹配

1. 输入img要修改为float()格式float32，否则跟weight不匹配报错
   输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错
   img = img.float()
   label = label.long()
   这两句要放在每个batch开始位置

   为了避免遗忘，可以把这部分操作集成到自定义的to_tensor()函数中，在每次开始转tensor的时候自动转换：
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


### 关于图片标签值的定义在分类问题和检测问题上的区别

1. 在纯分类任务中，数据集的label一般定义成从0开始，比如10类就是[0,9]，这样的好处是在转独热编码的时候比较容易，
   比如标签2的独热编码就是[0,0,1,0], 标签0的独热编码就是[1,0,0,0]
2. 而在物体检测任务中的分类子任务中，一般会把数据集的label定义成从1开始，比如10类就是[1,10], 这样做的目的是
   因为在检测任务中需要对anchor的身份进行指定，而比较简洁的处理是把负样本的anchor设定为label=0。所以相当于把
   label=0预留给anchor的负样本。


### 关于transform中涉及的类型变换导致的错误

1. transform和transform_inv中涉及的数据类型变换很多种类，很容易漏掉没有做而导致输出形式不对。

2. 对于transform_inv的变换，需要重点关注
    - 默认数据集输出类型：hwc, bgr。采用这种默认输出形式，主要是因为用opencv作为底层函数的输出就是这种形式。
      而pytorch需要的形式是chw, rgb，所以经过transform后输出就是chw,rgb
    - 逆变换需要先变换chw为hwc，然后才变换rgb为bgr：因为rgb2bgr是基于最后一个维度是c来写的。
    - 逆变换需要把



### 关于训练过程中图片尺寸如何变换的问题？

1. 通常会定义一个边框尺寸，比如scale = (300, 300)，这是图片的最大尺寸范围。

2. 图片首先经过transform，按比例缩放到边框尺寸，此时因为比例固定，所以每张图片尺寸都不同，但都有一条片跟边框尺寸拉平相等。
   比如一个batch的图片可能尺寸会变成(300, 256),(300, 284),(240,300)这样的形式。

3. 图片然后经过dataloader的collate_fn，对一个batch的图片取最大外沿，进行padding变成相同尺寸的图片。
   由于transform时所有图片已有一条边靠近边框尺寸，所以取所有图片最大外沿结果基本都是边框尺寸，比如一个batch的图片会变成
   (300,300),(300,300),(300,300)然后堆叠成(3,300,300), 所以最终进行训练的图片基本都是以边框尺寸进行训练。


### 关于训练时候的路径问题

1. 经常会产生路径找不到的情况，比较合理的解决方案是：
   - 数据集的root路径，尽可能采用绝对路径，即以/开头的绝对路径。
   - 项目的root路径尽可能加到sys.path中去。

2. 如果有报错说部分路径无法导入，一般有2种可能性：
   - 根目录路径不在sys.path中，可通过添加根目录路径到sys.path并结合根目录下一级目录可见的原则，实现大部分文件的导入。 
   - 被导入的module中含有错误的导入，需要修正这些错误，才能解决报错
   - 存在交叉导入：即A要导入B，同时B也要导入A，此时有可能产生错误，需要解除交叉导入才能解决报错。
   

### 关于归一化和标准化

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


### 关于卷积的取整方式

1. pytorch中conv, maxpool在计算输出w,h尺寸时，默认的取整方式都是下取整(floor)。
   唯一不同的是，maxpool可以手动设置成上取整即ceil mode，但conv不能手动设置，也就只能下取整。

2. conv, maxpool两者计算输出尺寸的公式一样
    - 没有dilation时，w' = (w - ksize + 2p)/s + 1
    - 有diliation时，相当于ksize被扩大，此时
                     w' = (w - d(ksize-1) +2p -1)/s  + 1
                     
                          
      
### 关于神经网络的前向计算和反向传播在pytorch中的对应

1. 前向传播：就是模型一层一层计算，output = model(img)
2. 损失计算：采用pytorch自带的nn.CrossEntropyLoss(y_pred, y_label)，则不需要手动增加softmax，
   也不需要手动把label转换成独热编码。因为这两部分都被写在pytorch自带的交叉熵函数对象内部了。
3. 损失反向传播：必须针对损失的标量进行，也就是losses先做规约缩减(reduction='mean')，然后才能
   loss.backward()，即这里loss是一个标量值，pytorch对这个标量内部梯度会自动获取，并反向传播。
4. 优化器更新权值：必须采用optimizer.step()完成
5. 额外要求：必须增加一句优化器梯度清零，optimizer.zero_grad()，这句必须放在backward之前，
   确保变量的梯度不会一直累加，而是每个batch独立计算，一个batch结束就清零一次。
   (自己写的神经网络，梯度是整个batch一起算，不需要累加，计算以后直接赋值，所以也就不需要清零了。) 



### 关于在GPU训练

1. 如果要在GPU训练，只需要3步
    - 创建设备device:  device = torch.
    - 模型送入device
    - batch data送入device: 注意这里其实只要img送入device就可以，因为跟model相关的计算只需要img输入

2. 并行式GPU训练并不一定比单GPU快，相反对于一些比较小的模型，单GPU的速度远超过并行式训练的速度。
   可能因为并行式训练需要让数据在GPU之间搬运造成时间损耗，同时python的并行式训练并不是真正的并行，
   而是在同一时间下只有一块GPU运行的假并行，只是能利用多GPU内存而不能利用多GPU算力的假并行。

3. 分布式训练才是真正的多GPU算力并行训练。


### 关于pytorch中几种不同类型模型的差异

1. model
2. model = model.to(device)
3. model = model.cuda()
3. model = nn.DataParallel(model)


### 关于如何设置DataLoader

1. 对常规的数据集，如果图片尺寸都是一样的，那么直接使用pytorch默认DataLoader就可以。
2. 对数据集中img的尺寸不一样的，由于dataloader需要对img进行堆叠，此时图片尺寸不同无法直接堆叠，
   则需要自定义collate_fn对img进行堆叠，同时处理那些labels/bboxes/segments是堆叠还是放入list.
   
3. pytorch默认的collate_fn设置不是None而是default_collate_fn，所以即使不用collate_fn
   选项，也不要去把collate_fn设置为None，这会导致collate_fn找不到可用的函数导致错误。
   (从这个角度说，pytorch的官方英文文档有问题，注明DataLoader的collate_fn默认=None，
   但实际上collate_fn的默认=default_collate_fn。)
   

### 关于contiguous的问题

1. numpy和pytorch都有可能产生in_contiguous的问题。主要都是在transpose(),permute()之后会发生。
参考：https://discuss.pytorch.org/t/negative-strides-of-numpy-array-with-torch-dataloader/28769

2. 在numpy数据下的报错形式是：ValueError: some of the strides of a given numpy array are negative. 
   This is currently not supported, but will be added in future releases.
   
3. 在tensor数据下的报错形式是：
   可通过t1.is_contiguous()查看到是否连续。

4. 解决方案;
    - 对于numpy：  img = np.ascontiguousarray(img)
    - 对于tensor： img = img.contiguous()


### 关于在dataset的__getitem__()中增加断点导致程序崩溃的问题

1. 现象：如果模型和数据送入GPU，dataloader会调用dataset的__getitem__函数获取数据进行堆叠，
   此时如果在__getitem__里边有自定义断点，会造成系统警告且暂停训练。
2. 解决方案：取消断点后模型训练/验证就正常了。而如果想要调试__getitem__里边的语法，可以设置
   额外的语句img, label = dataset[0]来进入__getitem__，或者next(iter(dataloader))


### 关于预训练模型的加载

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


### 关于全连接层的参数过多如何解决的问题

1. 全连接层所占参数一般占了一个模型总参数个数的80%以上，所以为了减小模型所占内存，需要替换全连接

2. 替换方式1：resnet/resnext的最后一部分就是这样实现。
   用一个adaptiveAvgpooling(1), 先把卷积的每个通道收缩为1，即(c,h,w)->(c,1,1)，再reshape后接一个简单全连接linear(c, 10)即可
   变换前reshape后需要至少2个全连接(c*h*w->256),再(256->10)，参数总量256*c*h*w + 256*10；
   avgpool变换后reshape后只需要1个全连接(256->10)，参数总量为256*10, 减少了256*c*h*w个参数。
   
   
   
   
### 关于整体模型设计的基本原则

0. 所有基本原则主要来自几个经典模型：vgg, resnet

1. 卷积负责增加层数，池化负责缩减尺寸：原则是尺寸缩减一倍时，层数增加一倍，这样能够保证特征的丰富性不受影响。

2. 也可以用卷积来增加层数，同时用卷积缩减尺寸

3. 主流卷积结构设置：
    - 两个3x3卷积串联：用来等效于5x5感受野的卷积核
    - 三个3x3卷积串联：用来等效于7x7感受野的卷积核
    - 1x1+3x3+1x3卷积串联结合恒等映射支路：基础残差模块用来多个串联且不容易模型性能退化(层数增加反而误差增加)
      这种结构有2个变体，一个变体用来增加层数(此时恒等映射支路有一个1x1)，另一个变体用来保持层数不变做叠加(此时恒等映射支路没有卷积)
    - 1x1+3x3+3x3+1x1卷积串联结合恒等映射支路


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
   通过


