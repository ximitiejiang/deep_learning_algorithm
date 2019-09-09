
# 临时记录pytorch训练深度神经网络的注意事项


### 关于数据格式

1. 默认的大部分数据集，输出格式都是n,h,w,c和bgr格式，一方面是hwc更普遍，另一方面是opencv读取的就是bgr。
2. pytorch中指定的数据格式是chw和rgb，所以常规处理方法是：数据集输出都统一定义成hwc和bgr，再通过
   transform来转换成chw和rgb
   
   
### 关于img/label与模型weight之间的数据格式匹配

1. 输入img要修改为float()格式float32，否则跟weight不匹配报错
   输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错
   img = img.float()
   label = label.long()
   这两句要放在每个batch开始位置。
   
   
### 关于训练时候的路径问题

1. 经常会产生路径找不到的情况，比较合理的解决方案是：
   - 数据集的root路径，尽可能采用绝对路径，即以/开头的绝对路径。
   - 项目的root路径尽可能加到sys.path中去。
   

### 关于归一化和标准化

1. 常见训练采用的mean, std参数(均来自mmdetection)：
   这些训练所采用的mean,std并不是跟训练数据集相关，而是跟基模型所训练的数据集相关，这是
   因为这些训练都是在基模型的基础上进行finetunning做迁移学习来训练的。
   由于pytorch的基模型基本都是在imagenet中训练的，所以这些mean, std都是imagenet的参数。
   而caffe的基模型虽然也是在imagenet中训练的，但因为处理方式不同所以std取成了1(待在caffe中确认原因)
   比如：
   - 来自pytorch的基模型：[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
   - 来自caffe的基模型：[123.675, 116.28, 103.53], std=[1, 1, 1]

2. 如果在一个数据集上从头训练，则需要事先计算他的mean, std。
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
      
      
### 关于神经网络的前向计算和反向传播在pytorch中的对应

1. 前向传播：就是模型一层一层计算，output = model(img)
2. 损失计算：采用pytorch自带的nn.CrossEntropyLoss(y_pred, y_label)，则不需要手动增加softmax，
   也不需要手动把label转换成独热编码。
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
    - batch data送入device

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

1. 对常规数据集，输出img, label，直接使用pytorch默认DataLoader就可以
2. 对数据集中需要输出除了img/label之外额外参数的，比如bbox, scale, shape，则需要
   自定义collate_fn对数据进行堆叠。
   
3. pytorch默认的collate_fn设置不是None而是default_collate_fn，所以即使不用collate_fn
   选项，也不要去把collate_fn设置为None，这会导致collate_fn找不到可用的函数导致错误。
   (从这个角度说，pytorch的官方英文文档有问题，注明DataLoader的collate_fn默认=None，
   但实际上collate_fn的默认=default_collate_fn。)
   

### 关于在dataset的__getitem__()中增加断点导致程序崩溃的问题

1. 现象：如果模型和数据送入GPU，dataloader会调用dataset的__getitem__函数获取数据进行堆叠，
   此时如果在__getitem__里边有自定义断点，会造成系统警告且暂停训练。
2. 解决方案：取消断点后模型训练/验证就正常了。而如果想要调试__getitem__里边的语法，可以设置
   额外的语句img, label = dataset[0]来进入__getitem__，或者next(iter(dataloader))



### 关于如何提升精度的tricks

1. 没有BN可以训练，但增加卷积之后激活之前的batchnorm2d()层，以及全连接之后激活之前的batchnorm1d()层，
   可至少增加10%以上的精度。
   
2. 没有初始化而采用pytorch默认初始化值可以训练，但增加恰当的初始化手段(xavier_init/ kaiming_init)，可提高精度

3. 固定学习率可以训练，但增加warmup的可变学习率，可提高精度

4. 训练精度如果已足够高，比如达到1或0.99，但验证精度不高，比如只有0.8，此时说明训练过拟合了。
   通过


