### 第一版ssd onnx, 发现有大量long类型，并且输出层数非常多，并且onnx模型无法生成engine

graph(%0 : Float(1, 3, 300, 300),
      %backbone.features.0.weight : Float(64, 3, 3, 3),
      %backbone.features.0.bias : Float(64),
      %backbone.features.2.weight : Float(64, 64, 3, 3),
      %backbone.features.2.bias : Float(64),
      %backbone.features.5.weight : Float(128, 64, 3, 3),
      %backbone.features.5.bias : Float(128),
      %backbone.features.7.weight : Float(128, 128, 3, 3),
      %backbone.features.7.bias : Float(128),
      %backbone.features.10.weight : Float(256, 128, 3, 3),
      %backbone.features.10.bias : Float(256),
      %backbone.features.12.weight : Float(256, 256, 3, 3),
      %backbone.features.12.bias : Float(256),
      %backbone.features.14.weight : Float(256, 256, 3, 3),
      %backbone.features.14.bias : Float(256),
      %backbone.features.17.weight : Float(512, 256, 3, 3),
      %backbone.features.17.bias : Float(512),
      %backbone.features.19.weight : Float(512, 512, 3, 3),
      %backbone.features.19.bias : Float(512),
      %backbone.features.21.weight : Float(512, 512, 3, 3),
      %backbone.features.21.bias : Float(512),
      %backbone.features.24.weight : Float(512, 512, 3, 3),
      %backbone.features.24.bias : Float(512),
      %backbone.features.26.weight : Float(512, 512, 3, 3),
      %backbone.features.26.bias : Float(512),
      %backbone.features.28.weight : Float(512, 512, 3, 3),
      %backbone.features.28.bias : Float(512),
      %backbone.features.31.weight : Float(1024, 512, 3, 3),
      %backbone.features.31.bias : Float(1024),
      %backbone.features.33.weight : Float(1024, 1024, 1, 1),
      %backbone.features.33.bias : Float(1024),
      %backbone.extra.0.weight : Float(256, 1024, 1, 1),
      %backbone.extra.0.bias : Float(256),
      %backbone.extra.1.weight : Float(512, 256, 3, 3),
      %backbone.extra.1.bias : Float(512),
      %backbone.extra.2.weight : Float(128, 512, 1, 1),
      %backbone.extra.2.bias : Float(128),
      %backbone.extra.3.weight : Float(256, 128, 3, 3),
      %backbone.extra.3.bias : Float(256),
      %backbone.extra.4.weight : Float(128, 256, 1, 1),
      %backbone.extra.4.bias : Float(128),
      %backbone.extra.5.weight : Float(256, 128, 3, 3),
      %backbone.extra.5.bias : Float(256),
      %backbone.extra.6.weight : Float(128, 256, 1, 1),
      %backbone.extra.6.bias : Float(128),
      %backbone.extra.7.weight : Float(256, 128, 3, 3),
      %backbone.extra.7.bias : Float(256),
      %backbone.l2_norm.weight : Float(512),
      %bbox_head.cls_convs.0.conv3x3.weight : Float(84, 512, 3, 3),
      %bbox_head.cls_convs.0.conv3x3.bias : Float(84),
      %bbox_head.cls_convs.1.conv3x3.weight : Float(126, 1024, 3, 3),
      %bbox_head.cls_convs.1.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.2.conv3x3.weight : Float(126, 512, 3, 3),
      %bbox_head.cls_convs.2.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.3.conv3x3.weight : Float(126, 256, 3, 3),
      %bbox_head.cls_convs.3.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.4.conv3x3.weight : Float(84, 256, 3, 3),
      %bbox_head.cls_convs.4.conv3x3.bias : Float(84),
      %bbox_head.cls_convs.5.conv3x3.weight : Float(84, 256, 3, 3),
      %bbox_head.cls_convs.5.conv3x3.bias : Float(84),
      %bbox_head.reg_convs.0.conv3x3.weight : Float(16, 512, 3, 3),
      %bbox_head.reg_convs.0.conv3x3.bias : Float(16),
      %bbox_head.reg_convs.1.conv3x3.weight : Float(24, 1024, 3, 3),
      %bbox_head.reg_convs.1.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.2.conv3x3.weight : Float(24, 512, 3, 3),
      %bbox_head.reg_convs.2.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.3.conv3x3.weight : Float(24, 256, 3, 3),
      %bbox_head.reg_convs.3.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.4.conv3x3.weight : Float(16, 256, 3, 3),
      %bbox_head.reg_convs.4.conv3x3.bias : Float(16),
      %bbox_head.reg_convs.5.conv3x3.weight : Float(16, 256, 3, 3),
      %bbox_head.reg_convs.5.conv3x3.bias : Float(16)):
  %72 : Float(1, 64, 300, 300) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%0, %backbone.features.0.weight, %backbone.features.0.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %73 : Float(1, 64, 300, 300) = onnx::Relu(%72), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %74 : Float(1, 64, 300, 300) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%73, %backbone.features.2.weight, %backbone.features.2.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %75 : Float(1, 64, 300, 300) = onnx::Relu(%74), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %76 : Float(1, 64, 150, 150) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%75), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %77 : Float(1, 128, 150, 150) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%76, %backbone.features.5.weight, %backbone.features.5.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %78 : Float(1, 128, 150, 150) = onnx::Relu(%77), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %79 : Float(1, 128, 150, 150) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%78, %backbone.features.7.weight, %backbone.features.7.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %80 : Float(1, 128, 150, 150) = onnx::Relu(%79), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %81 : Float(1, 128, 75, 75) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%80), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %82 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%81, %backbone.features.10.weight, %backbone.features.10.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %83 : Float(1, 256, 75, 75) = onnx::Relu(%82), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %84 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%83, %backbone.features.12.weight, %backbone.features.12.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %85 : Float(1, 256, 75, 75) = onnx::Relu(%84), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %86 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%85, %backbone.features.14.weight, %backbone.features.14.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %87 : Float(1, 256, 75, 75) = onnx::Relu(%86), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %88 : Float(1, 256, 38, 38) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%87), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %89 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%88, %backbone.features.17.weight, %backbone.features.17.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %90 : Float(1, 512, 38, 38) = onnx::Relu(%89), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %91 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%90, %backbone.features.19.weight, %backbone.features.19.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %92 : Float(1, 512, 38, 38) = onnx::Relu(%91), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %93 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%92, %backbone.features.21.weight, %backbone.features.21.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %94 : Float(1, 512, 38, 38) = onnx::Relu(%93), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %95 : Float(1, 512, 19, 19) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%94), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %96 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%95, %backbone.features.24.weight, %backbone.features.24.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %97 : Float(1, 512, 19, 19) = onnx::Relu(%96), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %98 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%97, %backbone.features.26.weight, %backbone.features.26.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %99 : Float(1, 512, 19, 19) = onnx::Relu(%98), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %100 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%99, %backbone.features.28.weight, %backbone.features.28.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %101 : Float(1, 512, 19, 19) = onnx::Relu(%100), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %102 : Float(1, 512, 19, 19) = onnx::MaxPool[kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%101), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %103 : Float(1, 1024, 19, 19) = onnx::Conv[dilations=[6, 6], group=1, kernel_shape=[3, 3], pads=[6, 6, 6, 6], strides=[1, 1]](%102, %backbone.features.31.weight, %backbone.features.31.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %104 : Float(1, 1024, 19, 19) = onnx::Relu(%103), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %105 : Float(1, 1024, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%104, %backbone.features.33.weight, %backbone.features.33.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %106 : Float(1, 1024, 19, 19) = onnx::Relu(%105), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %107 : Float(1, 256, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%106, %backbone.extra.0.weight, %backbone.extra.0.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %108 : Float(1, 256, 19, 19) = onnx::Relu(%107), scope: OneStageDetector/SSDVGG16[backbone]
  %109 : Float(1, 512, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]](%108, %backbone.extra.1.weight, %backbone.extra.1.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %110 : Float(1, 512, 10, 10) = onnx::Relu(%109), scope: OneStageDetector/SSDVGG16[backbone]
  %111 : Float(1, 128, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%110, %backbone.extra.2.weight, %backbone.extra.2.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %112 : Float(1, 128, 10, 10) = onnx::Relu(%111), scope: OneStageDetector/SSDVGG16[backbone]
  %113 : Float(1, 256, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]](%112, %backbone.extra.3.weight, %backbone.extra.3.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %114 : Float(1, 256, 5, 5) = onnx::Relu(%113), scope: OneStageDetector/SSDVGG16[backbone]
  %115 : Float(1, 128, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%114, %backbone.extra.4.weight, %backbone.extra.4.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %116 : Float(1, 128, 5, 5) = onnx::Relu(%115), scope: OneStageDetector/SSDVGG16[backbone]
  %117 : Float(1, 256, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%116, %backbone.extra.5.weight, %backbone.extra.5.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %118 : Float(1, 256, 3, 3) = onnx::Relu(%117), scope: OneStageDetector/SSDVGG16[backbone]
  %119 : Float(1, 128, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%118, %backbone.extra.6.weight, %backbone.extra.6.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %120 : Float(1, 128, 3, 3) = onnx::Relu(%119), scope: OneStageDetector/SSDVGG16[backbone]
  %121 : Float(1, 256, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%120, %backbone.extra.7.weight, %backbone.extra.7.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %122 : Float(1, 256, 1, 1) = onnx::Relu(%121), scope: OneStageDetector/SSDVGG16[backbone]
  %123 : Tensor = onnx::Constant[value={2}](), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %124 : Float(1, 512, 38, 38) = onnx::Pow(%94, %123), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %125 : Float(1, 1, 38, 38) = onnx::ReduceSum[axes=[1], keepdims=1](%124), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %126 : Float(1, 1, 38, 38) = onnx::Sqrt(%125), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %127 : Tensor = onnx::Constant[value={1e-10}]()
  %128 : Tensor = onnx::Add(%126, %127)
  %129 : Float(1, 512) = onnx::Unsqueeze[axes=[0]](%backbone.l2_norm.weight), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %130 : Float(1, 512, 1) = onnx::Unsqueeze[axes=[2]](%129), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %131 : Float(1, 512, 1, 1) = onnx::Unsqueeze[axes=[3]](%130), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %132 : Tensor = onnx::Shape(%94), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %133 : Float(1, 512!, 38, 38!) = onnx::Expand(%131, %132), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %134 : Float(1, 512, 38, 38) = onnx::Mul(%133, %94), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %135 : Float(1, 512, 38, 38) = onnx::Div(%134, %128), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %136 : Float(1, 84, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%135, %bbox_head.cls_convs.0.conv3x3.weight, %bbox_head.cls_convs.0.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %137 : Float(1, 38, 38, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%136), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %138 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %139 : Tensor = onnx::Shape(%137), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %140 : Long() = onnx::Gather[axis=0](%139, %138), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %141 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %142 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %143 : Tensor = onnx::Unsqueeze[axes=[0]](%140)
  %144 : Tensor = onnx::Unsqueeze[axes=[0]](%141)
  %145 : Tensor = onnx::Unsqueeze[axes=[0]](%142)
  %146 : Tensor = onnx::Concat[axis=0](%143, %144, %145)
  %147 : Float(1, 5776, 21) = onnx::Reshape(%137, %146), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %148 : Float(1, 16, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%135, %bbox_head.reg_convs.0.conv3x3.weight, %bbox_head.reg_convs.0.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %149 : Float(1, 38, 38, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%148), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %150 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %151 : Tensor = onnx::Shape(%149), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %152 : Long() = onnx::Gather[axis=0](%151, %150), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %153 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %154 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %155 : Tensor = onnx::Unsqueeze[axes=[0]](%152)
  %156 : Tensor = onnx::Unsqueeze[axes=[0]](%153)
  %157 : Tensor = onnx::Unsqueeze[axes=[0]](%154)
  %158 : Tensor = onnx::Concat[axis=0](%155, %156, %157)
  %159 : Float(1, 5776, 4) = onnx::Reshape(%149, %158), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %160 : Float(1, 126, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%106, %bbox_head.cls_convs.1.conv3x3.weight, %bbox_head.cls_convs.1.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %161 : Float(1, 19, 19, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%160), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %162 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %163 : Tensor = onnx::Shape(%161), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %164 : Long() = onnx::Gather[axis=0](%163, %162), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %165 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %166 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %167 : Tensor = onnx::Unsqueeze[axes=[0]](%164)
  %168 : Tensor = onnx::Unsqueeze[axes=[0]](%165)
  %169 : Tensor = onnx::Unsqueeze[axes=[0]](%166)
  %170 : Tensor = onnx::Concat[axis=0](%167, %168, %169)
  %171 : Float(1, 2166, 21) = onnx::Reshape(%161, %170), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %172 : Float(1, 24, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%106, %bbox_head.reg_convs.1.conv3x3.weight, %bbox_head.reg_convs.1.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %173 : Float(1, 19, 19, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%172), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %174 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %175 : Tensor = onnx::Shape(%173), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %176 : Long() = onnx::Gather[axis=0](%175, %174), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %177 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %178 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %179 : Tensor = onnx::Unsqueeze[axes=[0]](%176)
  %180 : Tensor = onnx::Unsqueeze[axes=[0]](%177)
  %181 : Tensor = onnx::Unsqueeze[axes=[0]](%178)
  %182 : Tensor = onnx::Concat[axis=0](%179, %180, %181)
  %183 : Float(1, 2166, 4) = onnx::Reshape(%173, %182), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %184 : Float(1, 126, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%110, %bbox_head.cls_convs.2.conv3x3.weight, %bbox_head.cls_convs.2.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %185 : Float(1, 10, 10, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%184), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %186 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %187 : Tensor = onnx::Shape(%185), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %188 : Long() = onnx::Gather[axis=0](%187, %186), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %189 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %190 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %191 : Tensor = onnx::Unsqueeze[axes=[0]](%188)
  %192 : Tensor = onnx::Unsqueeze[axes=[0]](%189)
  %193 : Tensor = onnx::Unsqueeze[axes=[0]](%190)
  %194 : Tensor = onnx::Concat[axis=0](%191, %192, %193)
  %195 : Float(1, 600, 21) = onnx::Reshape(%185, %194), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %196 : Float(1, 24, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%110, %bbox_head.reg_convs.2.conv3x3.weight, %bbox_head.reg_convs.2.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %197 : Float(1, 10, 10, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%196), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %198 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %199 : Tensor = onnx::Shape(%197), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %200 : Long() = onnx::Gather[axis=0](%199, %198), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %201 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %202 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %203 : Tensor = onnx::Unsqueeze[axes=[0]](%200)
  %204 : Tensor = onnx::Unsqueeze[axes=[0]](%201)
  %205 : Tensor = onnx::Unsqueeze[axes=[0]](%202)
  %206 : Tensor = onnx::Concat[axis=0](%203, %204, %205)
  %207 : Float(1, 600, 4) = onnx::Reshape(%197, %206), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %208 : Float(1, 126, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%114, %bbox_head.cls_convs.3.conv3x3.weight, %bbox_head.cls_convs.3.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %209 : Float(1, 5, 5, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%208), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %210 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %211 : Tensor = onnx::Shape(%209), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %212 : Long() = onnx::Gather[axis=0](%211, %210), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %213 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %214 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %215 : Tensor = onnx::Unsqueeze[axes=[0]](%212)
  %216 : Tensor = onnx::Unsqueeze[axes=[0]](%213)
  %217 : Tensor = onnx::Unsqueeze[axes=[0]](%214)
  %218 : Tensor = onnx::Concat[axis=0](%215, %216, %217)
  %219 : Float(1, 150, 21) = onnx::Reshape(%209, %218), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %220 : Float(1, 24, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%114, %bbox_head.reg_convs.3.conv3x3.weight, %bbox_head.reg_convs.3.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %221 : Float(1, 5, 5, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%220), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %222 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %223 : Tensor = onnx::Shape(%221), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %224 : Long() = onnx::Gather[axis=0](%223, %222), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %225 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %226 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %227 : Tensor = onnx::Unsqueeze[axes=[0]](%224)
  %228 : Tensor = onnx::Unsqueeze[axes=[0]](%225)
  %229 : Tensor = onnx::Unsqueeze[axes=[0]](%226)
  %230 : Tensor = onnx::Concat[axis=0](%227, %228, %229)
  %231 : Float(1, 150, 4) = onnx::Reshape(%221, %230), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %232 : Float(1, 84, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%118, %bbox_head.cls_convs.4.conv3x3.weight, %bbox_head.cls_convs.4.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %233 : Float(1, 3, 3, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%232), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %234 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %235 : Tensor = onnx::Shape(%233), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %236 : Long() = onnx::Gather[axis=0](%235, %234), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %237 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %238 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %239 : Tensor = onnx::Unsqueeze[axes=[0]](%236)
  %240 : Tensor = onnx::Unsqueeze[axes=[0]](%237)
  %241 : Tensor = onnx::Unsqueeze[axes=[0]](%238)
  %242 : Tensor = onnx::Concat[axis=0](%239, %240, %241)
  %243 : Float(1, 36, 21) = onnx::Reshape(%233, %242), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %244 : Float(1, 16, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%118, %bbox_head.reg_convs.4.conv3x3.weight, %bbox_head.reg_convs.4.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %245 : Float(1, 3, 3, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%244), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %246 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %247 : Tensor = onnx::Shape(%245), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %248 : Long() = onnx::Gather[axis=0](%247, %246), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %249 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %250 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %251 : Tensor = onnx::Unsqueeze[axes=[0]](%248)
  %252 : Tensor = onnx::Unsqueeze[axes=[0]](%249)
  %253 : Tensor = onnx::Unsqueeze[axes=[0]](%250)
  %254 : Tensor = onnx::Concat[axis=0](%251, %252, %253)
  %255 : Float(1, 36, 4) = onnx::Reshape(%245, %254), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %256 : Float(1, 84, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%122, %bbox_head.cls_convs.5.conv3x3.weight, %bbox_head.cls_convs.5.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %257 : Float(1!, 1, 1!, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%256), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %258 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %259 : Tensor = onnx::Shape(%257), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %260 : Long() = onnx::Gather[axis=0](%259, %258), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %261 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %262 : Long() = onnx::Constant[value={21}](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %263 : Tensor = onnx::Unsqueeze[axes=[0]](%260)
  %264 : Tensor = onnx::Unsqueeze[axes=[0]](%261)
  %265 : Tensor = onnx::Unsqueeze[axes=[0]](%262)
  %266 : Tensor = onnx::Concat[axis=0](%263, %264, %265)
  %267 : Float(1, 4, 21) = onnx::Reshape(%257, %266), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %268 : Float(1, 16, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%122, %bbox_head.reg_convs.5.conv3x3.weight, %bbox_head.reg_convs.5.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %269 : Float(1!, 1, 1!, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%268), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %270 : Long() = onnx::Constant[value={0}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %271 : Tensor = onnx::Shape(%269), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %272 : Long() = onnx::Gather[axis=0](%271, %270), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %273 : Long() = onnx::Constant[value={-1}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %274 : Long() = onnx::Constant[value={4}](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %275 : Tensor = onnx::Unsqueeze[axes=[0]](%272)
  %276 : Tensor = onnx::Unsqueeze[axes=[0]](%273)
  %277 : Tensor = onnx::Unsqueeze[axes=[0]](%274)
  %278 : Tensor = onnx::Concat[axis=0](%275, %276, %277)
  %279 : Float(1, 4, 4) = onnx::Reshape(%269, %278), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %280 : Float(1, 8732, 21) = onnx::Concat[axis=1](%147, %171, %195, %219, %243, %267), scope: OneStageDetector/SSDHead[bbox_head]
  %281 : Float(1, 8732, 4) = onnx::Concat[axis=1](%159, %183, %207, %231, %255, %279), scope: OneStageDetector/SSDHead[bbox_head]
  return (%280, %281)
  
### 这是第二版ssd onnx， 把cls_head, bbox_head里边的view语句中常量都增加强制转换为view(int(out.size(0)), int(-1), int(self.num_classes))，层数从281变为185！
graph(%0 : Float(1, 3, 300, 300),
      %backbone.features.0.weight : Float(64, 3, 3, 3),
      %backbone.features.0.bias : Float(64),
      %backbone.features.2.weight : Float(64, 64, 3, 3),
      %backbone.features.2.bias : Float(64),
      %backbone.features.5.weight : Float(128, 64, 3, 3),
      %backbone.features.5.bias : Float(128),
      %backbone.features.7.weight : Float(128, 128, 3, 3),
      %backbone.features.7.bias : Float(128),
      %backbone.features.10.weight : Float(256, 128, 3, 3),
      %backbone.features.10.bias : Float(256),
      %backbone.features.12.weight : Float(256, 256, 3, 3),
      %backbone.features.12.bias : Float(256),
      %backbone.features.14.weight : Float(256, 256, 3, 3),
      %backbone.features.14.bias : Float(256),
      %backbone.features.17.weight : Float(512, 256, 3, 3),
      %backbone.features.17.bias : Float(512),
      %backbone.features.19.weight : Float(512, 512, 3, 3),
      %backbone.features.19.bias : Float(512),
      %backbone.features.21.weight : Float(512, 512, 3, 3),
      %backbone.features.21.bias : Float(512),
      %backbone.features.24.weight : Float(512, 512, 3, 3),
      %backbone.features.24.bias : Float(512),
      %backbone.features.26.weight : Float(512, 512, 3, 3),
      %backbone.features.26.bias : Float(512),
      %backbone.features.28.weight : Float(512, 512, 3, 3),
      %backbone.features.28.bias : Float(512),
      %backbone.features.31.weight : Float(1024, 512, 3, 3),
      %backbone.features.31.bias : Float(1024),
      %backbone.features.33.weight : Float(1024, 1024, 1, 1),
      %backbone.features.33.bias : Float(1024),
      %backbone.extra.0.weight : Float(256, 1024, 1, 1),
      %backbone.extra.0.bias : Float(256),
      %backbone.extra.1.weight : Float(512, 256, 3, 3),
      %backbone.extra.1.bias : Float(512),
      %backbone.extra.2.weight : Float(128, 512, 1, 1),
      %backbone.extra.2.bias : Float(128),
      %backbone.extra.3.weight : Float(256, 128, 3, 3),
      %backbone.extra.3.bias : Float(256),
      %backbone.extra.4.weight : Float(128, 256, 1, 1),
      %backbone.extra.4.bias : Float(128),
      %backbone.extra.5.weight : Float(256, 128, 3, 3),
      %backbone.extra.5.bias : Float(256),
      %backbone.extra.6.weight : Float(128, 256, 1, 1),
      %backbone.extra.6.bias : Float(128),
      %backbone.extra.7.weight : Float(256, 128, 3, 3),
      %backbone.extra.7.bias : Float(256),
      %backbone.l2_norm.weight : Float(512),
      %bbox_head.cls_convs.0.conv3x3.weight : Float(84, 512, 3, 3),
      %bbox_head.cls_convs.0.conv3x3.bias : Float(84),
      %bbox_head.cls_convs.1.conv3x3.weight : Float(126, 1024, 3, 3),
      %bbox_head.cls_convs.1.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.2.conv3x3.weight : Float(126, 512, 3, 3),
      %bbox_head.cls_convs.2.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.3.conv3x3.weight : Float(126, 256, 3, 3),
      %bbox_head.cls_convs.3.conv3x3.bias : Float(126),
      %bbox_head.cls_convs.4.conv3x3.weight : Float(84, 256, 3, 3),
      %bbox_head.cls_convs.4.conv3x3.bias : Float(84),
      %bbox_head.cls_convs.5.conv3x3.weight : Float(84, 256, 3, 3),
      %bbox_head.cls_convs.5.conv3x3.bias : Float(84),
      %bbox_head.reg_convs.0.conv3x3.weight : Float(16, 512, 3, 3),
      %bbox_head.reg_convs.0.conv3x3.bias : Float(16),
      %bbox_head.reg_convs.1.conv3x3.weight : Float(24, 1024, 3, 3),
      %bbox_head.reg_convs.1.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.2.conv3x3.weight : Float(24, 512, 3, 3),
      %bbox_head.reg_convs.2.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.3.conv3x3.weight : Float(24, 256, 3, 3),
      %bbox_head.reg_convs.3.conv3x3.bias : Float(24),
      %bbox_head.reg_convs.4.conv3x3.weight : Float(16, 256, 3, 3),
      %bbox_head.reg_convs.4.conv3x3.bias : Float(16),
      %bbox_head.reg_convs.5.conv3x3.weight : Float(16, 256, 3, 3),
      %bbox_head.reg_convs.5.conv3x3.bias : Float(16)):
  %72 : Float(1, 64, 300, 300) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%0, %backbone.features.0.weight, %backbone.features.0.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %73 : Float(1, 64, 300, 300) = onnx::Relu(%72), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %74 : Float(1, 64, 300, 300) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%73, %backbone.features.2.weight, %backbone.features.2.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %75 : Float(1, 64, 300, 300) = onnx::Relu(%74), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %76 : Float(1, 64, 150, 150) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%75), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %77 : Float(1, 128, 150, 150) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%76, %backbone.features.5.weight, %backbone.features.5.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %78 : Float(1, 128, 150, 150) = onnx::Relu(%77), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %79 : Float(1, 128, 150, 150) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%78, %backbone.features.7.weight, %backbone.features.7.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %80 : Float(1, 128, 150, 150) = onnx::Relu(%79), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %81 : Float(1, 128, 75, 75) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%80), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %82 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%81, %backbone.features.10.weight, %backbone.features.10.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %83 : Float(1, 256, 75, 75) = onnx::Relu(%82), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %84 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%83, %backbone.features.12.weight, %backbone.features.12.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %85 : Float(1, 256, 75, 75) = onnx::Relu(%84), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %86 : Float(1, 256, 75, 75) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%85, %backbone.features.14.weight, %backbone.features.14.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %87 : Float(1, 256, 75, 75) = onnx::Relu(%86), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %88 : Float(1, 256, 38, 38) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%87), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %89 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%88, %backbone.features.17.weight, %backbone.features.17.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %90 : Float(1, 512, 38, 38) = onnx::Relu(%89), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %91 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%90, %backbone.features.19.weight, %backbone.features.19.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %92 : Float(1, 512, 38, 38) = onnx::Relu(%91), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %93 : Float(1, 512, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%92, %backbone.features.21.weight, %backbone.features.21.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %94 : Float(1, 512, 38, 38) = onnx::Relu(%93), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %95 : Float(1, 512, 19, 19) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[2, 2]](%94), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %96 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%95, %backbone.features.24.weight, %backbone.features.24.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %97 : Float(1, 512, 19, 19) = onnx::Relu(%96), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %98 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%97, %backbone.features.26.weight, %backbone.features.26.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %99 : Float(1, 512, 19, 19) = onnx::Relu(%98), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %100 : Float(1, 512, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%99, %backbone.features.28.weight, %backbone.features.28.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %101 : Float(1, 512, 19, 19) = onnx::Relu(%100), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %102 : Float(1, 512, 19, 19) = onnx::MaxPool[kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%101), scope: OneStageDetector/SSDVGG16[backbone]/MaxPool2d
  %103 : Float(1, 1024, 19, 19) = onnx::Conv[dilations=[6, 6], group=1, kernel_shape=[3, 3], pads=[6, 6, 6, 6], strides=[1, 1]](%102, %backbone.features.31.weight, %backbone.features.31.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %104 : Float(1, 1024, 19, 19) = onnx::Relu(%103), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %105 : Float(1, 1024, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%104, %backbone.features.33.weight, %backbone.features.33.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %106 : Float(1, 1024, 19, 19) = onnx::Relu(%105), scope: OneStageDetector/SSDVGG16[backbone]/ReLU
  %107 : Float(1, 256, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%106, %backbone.extra.0.weight, %backbone.extra.0.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %108 : Float(1, 256, 19, 19) = onnx::Relu(%107), scope: OneStageDetector/SSDVGG16[backbone]
  %109 : Float(1, 512, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]](%108, %backbone.extra.1.weight, %backbone.extra.1.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %110 : Float(1, 512, 10, 10) = onnx::Relu(%109), scope: OneStageDetector/SSDVGG16[backbone]
  %111 : Float(1, 128, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%110, %backbone.extra.2.weight, %backbone.extra.2.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %112 : Float(1, 128, 10, 10) = onnx::Relu(%111), scope: OneStageDetector/SSDVGG16[backbone]
  %113 : Float(1, 256, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]](%112, %backbone.extra.3.weight, %backbone.extra.3.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %114 : Float(1, 256, 5, 5) = onnx::Relu(%113), scope: OneStageDetector/SSDVGG16[backbone]
  %115 : Float(1, 128, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%114, %backbone.extra.4.weight, %backbone.extra.4.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %116 : Float(1, 128, 5, 5) = onnx::Relu(%115), scope: OneStageDetector/SSDVGG16[backbone]
  %117 : Float(1, 256, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%116, %backbone.extra.5.weight, %backbone.extra.5.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %118 : Float(1, 256, 3, 3) = onnx::Relu(%117), scope: OneStageDetector/SSDVGG16[backbone]
  %119 : Float(1, 128, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%118, %backbone.extra.6.weight, %backbone.extra.6.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %120 : Float(1, 128, 3, 3) = onnx::Relu(%119), scope: OneStageDetector/SSDVGG16[backbone]
  %121 : Float(1, 256, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%120, %backbone.extra.7.weight, %backbone.extra.7.bias), scope: OneStageDetector/SSDVGG16[backbone]/Conv2d
  %122 : Float(1, 256, 1, 1) = onnx::Relu(%121), scope: OneStageDetector/SSDVGG16[backbone]
  %123 : Tensor = onnx::Constant[value={2}](), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %124 : Float(1, 512, 38, 38) = onnx::Pow(%94, %123), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %125 : Float(1, 1, 38, 38) = onnx::ReduceSum[axes=[1], keepdims=1](%124), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %126 : Float(1, 1, 38, 38) = onnx::Sqrt(%125), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %127 : Tensor = onnx::Constant[value={1e-10}]()
  %128 : Tensor = onnx::Add(%126, %127)
  %129 : Float(1, 512) = onnx::Unsqueeze[axes=[0]](%backbone.l2_norm.weight), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %130 : Float(1, 512, 1) = onnx::Unsqueeze[axes=[2]](%129), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %131 : Float(1, 512, 1, 1) = onnx::Unsqueeze[axes=[3]](%130), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %132 : Tensor = onnx::Shape(%94), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %133 : Float(1, 512!, 38, 38!) = onnx::Expand(%131, %132), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %134 : Float(1, 512, 38, 38) = onnx::Mul(%133, %94), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %135 : Float(1, 512, 38, 38) = onnx::Div(%134, %128), scope: OneStageDetector/SSDVGG16[backbone]/L2Norm[l2_norm]
  %136 : Float(1, 84, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%135, %bbox_head.cls_convs.0.conv3x3.weight, %bbox_head.cls_convs.0.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %137 : Float(1, 38, 38, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%136), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %138 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %139 : Float(1, 5776, 21) = onnx::Reshape(%137, %138), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %140 : Float(1, 16, 38, 38) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%135, %bbox_head.reg_convs.0.conv3x3.weight, %bbox_head.reg_convs.0.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %141 : Float(1, 38, 38, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%140), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %142 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %143 : Float(1, 5776, 4) = onnx::Reshape(%141, %142), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %144 : Float(1, 126, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%106, %bbox_head.cls_convs.1.conv3x3.weight, %bbox_head.cls_convs.1.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %145 : Float(1, 19, 19, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%144), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %146 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %147 : Float(1, 2166, 21) = onnx::Reshape(%145, %146), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %148 : Float(1, 24, 19, 19) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%106, %bbox_head.reg_convs.1.conv3x3.weight, %bbox_head.reg_convs.1.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %149 : Float(1, 19, 19, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%148), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %150 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %151 : Float(1, 2166, 4) = onnx::Reshape(%149, %150), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %152 : Float(1, 126, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%110, %bbox_head.cls_convs.2.conv3x3.weight, %bbox_head.cls_convs.2.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %153 : Float(1, 10, 10, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%152), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %154 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %155 : Float(1, 600, 21) = onnx::Reshape(%153, %154), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %156 : Float(1, 24, 10, 10) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%110, %bbox_head.reg_convs.2.conv3x3.weight, %bbox_head.reg_convs.2.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %157 : Float(1, 10, 10, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%156), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %158 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %159 : Float(1, 600, 4) = onnx::Reshape(%157, %158), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %160 : Float(1, 126, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%114, %bbox_head.cls_convs.3.conv3x3.weight, %bbox_head.cls_convs.3.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %161 : Float(1, 5, 5, 126) = onnx::Transpose[perm=[0, 2, 3, 1]](%160), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %162 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %163 : Float(1, 150, 21) = onnx::Reshape(%161, %162), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %164 : Float(1, 24, 5, 5) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%114, %bbox_head.reg_convs.3.conv3x3.weight, %bbox_head.reg_convs.3.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %165 : Float(1, 5, 5, 24) = onnx::Transpose[perm=[0, 2, 3, 1]](%164), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %166 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %167 : Float(1, 150, 4) = onnx::Reshape(%165, %166), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %168 : Float(1, 84, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%118, %bbox_head.cls_convs.4.conv3x3.weight, %bbox_head.cls_convs.4.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %169 : Float(1, 3, 3, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%168), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %170 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %171 : Float(1, 36, 21) = onnx::Reshape(%169, %170), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %172 : Float(1, 16, 3, 3) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%118, %bbox_head.reg_convs.4.conv3x3.weight, %bbox_head.reg_convs.4.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %173 : Float(1, 3, 3, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%172), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %174 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %175 : Float(1, 36, 4) = onnx::Reshape(%173, %174), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %176 : Float(1, 84, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%122, %bbox_head.cls_convs.5.conv3x3.weight, %bbox_head.cls_convs.5.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead/Conv2d[conv3x3]
  %177 : Float(1!, 1, 1!, 84) = onnx::Transpose[perm=[0, 2, 3, 1]](%176), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %178 : Tensor = onnx::Constant[value=  1  -1  21 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %179 : Float(1, 4, 21) = onnx::Reshape(%177, %178), scope: OneStageDetector/SSDHead[bbox_head]/ClassHead
  %180 : Float(1, 16, 1, 1) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%122, %bbox_head.reg_convs.5.conv3x3.weight, %bbox_head.reg_convs.5.conv3x3.bias), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead/Conv2d[conv3x3]
  %181 : Float(1!, 1, 1!, 16) = onnx::Transpose[perm=[0, 2, 3, 1]](%180), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %182 : Tensor = onnx::Constant[value= 1 -1  4 [ Variable[CPUType]{3} ]](), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %183 : Float(1, 4, 4) = onnx::Reshape(%181, %182), scope: OneStageDetector/SSDHead[bbox_head]/BboxHead
  %184 : Float(1, 8732, 21) = onnx::Concat[axis=1](%139, %147, %155, %163, %171, %179), scope: OneStageDetector/SSDHead[bbox_head]
  %185 : Float(1, 8732, 4) = onnx::Concat[axis=1](%143, %151, %159, %167, %175, %183), scope: OneStageDetector/SSDHead[bbox_head]
  return (%184, %185)  