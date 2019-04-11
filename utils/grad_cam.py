"""refer from: https://github.com/jacobgil/pytorch-grad-cam

显示每一层特征激活情况
"""

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
import cv2
import numpy as np

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    用于计算特征层的计算输出, 同时获得输出tensor的梯度值
    Args:
        model: 特征层模型
        target_layers(str): 目标层的名称
    Return:
        outputs(list): [t1,..tn] 目标层的输出，如果有多个目标层也都存放在一个list
        x(tensor): (b,c,h,w) 最后一层
        隐性输出gradients(list): [grad_t1,...grad_tn]输出层tensor的梯度,每一层一个grad
    """
    def __init__(self, model, target_layers):
        self.model = model    # 特征提取层
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)   # 计算特征层每一层的输出
            if name in self.target_layers:
                x.register_hook(self.save_gradient)  # 如果目标层名称符合，则注册一个hook函数save_gradient,注意pytorch的hook函数必须是func(grad)业绩是会传入这个tensor变量的grad,返回跟这个grad相关的任何处理，比如传入print这个内置函数，则会打印出来这个tensor的grad
                outputs += [x]                       # 如果目标层名称符合，则保存目标层输出
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. 
    """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(
            self.model.features, target_layers)  # 这里要求模型要有一个名叫features的module_list

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)  # 所有目标层输出，以及
		output = output.view(output.size(0), -1)  # (b,c,h,w)->(b, c*h*w)
		output = self.model.classifier(output)    # (b, c*h*w)->(b,n_class)
		return target_activations, output    

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
    """主模块：用于从图片中提取出指定层 的特征(如果没有指定则提取出score最高的类)
        1. 创建了extractor,
    """
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model   # 传入模型
        self.model.eval()    # 模型在测试模式
        self.cuda = use_cuda 
        if self.cuda:
            self.model = model.cuda()
            
        self.extractor = ModelOutputs(self.model, target_layer_names) # 创建模型解出类：指定模型，和目标层的名字

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())  # 输出经过feature layer, classifier layer后的tensor
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())   # 把n_class中最大的分类score的index提取出来，比如70就表示第70类

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1  # 转化为得分最高的那个类的score
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)  # 这里用两个概率相乘，相当于计算了损失，
                                              # 也就是nll loss的概念： 两个概率相乘求和
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)  # retain_variables已取消，这里改为retain_graph， 这里的one_hot相当与nll loss的反向传播
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 通过hook获的梯度，这里提取的是最后一层

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]   # (1,c,h,w)->(c,h,w)

        weights = np.mean(grads_val, axis = (2, 3))[0, :]  # 计算梯度均值，每层一个均值 grads_val (1,512,14,14) -> mean(axis=(2,3)) get (1,512)
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)  # (512,14,14) -> (14,14)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]   # cam = cam + 梯度均值×target  (14,14)相当与该层特征每个位置乘以平均梯度

        cam = np.maximum(cam, 0)           # ? 
        cam = cv2.resize(cam, (224, 224))  # 把得到的特征尺寸放大到原图
        cam = cam - np.min(cam)            
        cam = cam / np.max(cam)            # 进行归一化？
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

#def get_args():
#	parser = argparse.ArgumentParser()
#	parser.add_argument('--use-cuda', action='store_true', default=True,
#	                    help='Use NVIDIA GPU acceleration')
#	parser.add_argument('--image-path', type=str, default='./examples/141597.jpg',
#	                    help='Input image path')
#	args = parser.parse_args()
#	args.use_cuda = args.use_cuda and torch.cuda.is_available()
#	if args.use_cuda:
#	    print("Using GPU for acceleration")
#	else:
#	    print("Using CPU for computation")
#
#	return args
""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. 
"""
if __name__ == '__main__':
    use_cuda = True
    img_path = './examples/141597.jpg'
    

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
    """stpe1: 创建grad_cam对象，传入model/layer, 但要确保model里边有model.feature, model.classifier"""
    grad_cam = GradCam(model = models.vgg19(pretrained=True), \
                       target_layer_names = ["35"], use_cuda=use_cuda)
    
    """step2: 图片预处理，"""
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255  # 变更尺寸，归一化
    input = preprocess_image(img)  # (1, c, h, w)

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
    target_index = [1,3]   # 输入None则表示希望返回得分最高的类别，也可输入类的index，比如[2,5,12]就是希望输出第2,5,12类的特征映射情况
    mask = grad_cam(input, target_index)    
    
    show_cam_on_image(img, mask)
    
    # 把源模型中的ReLU模型替换为自定义的gbReLU, 用于生成另一种图（不是热图，暂时不研究）
#    gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=use_cuda)
#    gb = gb_model(input, index=target_index)
#    utils.save_image(torch.from_numpy(gb), 'gb.jpg')
#
#    cam_mask = np.zeros(gb.shape)
#    for i in range(0, gb.shape[0]):
#        cam_mask[i, :, :] = mask
#
#    cam_gb = np.multiply(cam_mask, gb)
#    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
    