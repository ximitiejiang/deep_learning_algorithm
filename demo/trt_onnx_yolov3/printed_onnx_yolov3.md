graph YOLOv3-608 (
  %000_net[FLOAT, 64x3x608x608]
) optional inputs with matching initializers (
  %001_convolutional_bn_scale[FLOAT, 32]
  %001_convolutional_bn_bias[FLOAT, 32]
  %001_convolutional_bn_mean[FLOAT, 32]
  %001_convolutional_bn_var[FLOAT, 32]
  %001_convolutional_conv_weights[FLOAT, 32x3x3x3]
  %002_convolutional_bn_scale[FLOAT, 64]
  %002_convolutional_bn_bias[FLOAT, 64]
  %002_convolutional_bn_mean[FLOAT, 64]
  %002_convolutional_bn_var[FLOAT, 64]
  %002_convolutional_conv_weights[FLOAT, 64x32x3x3]
  %003_convolutional_bn_scale[FLOAT, 32]
  %003_convolutional_bn_bias[FLOAT, 32]
  %003_convolutional_bn_mean[FLOAT, 32]
  %003_convolutional_bn_var[FLOAT, 32]
  %003_convolutional_conv_weights[FLOAT, 32x64x1x1]
  %004_convolutional_bn_scale[FLOAT, 64]
  %004_convolutional_bn_bias[FLOAT, 64]
  %004_convolutional_bn_mean[FLOAT, 64]
  %004_convolutional_bn_var[FLOAT, 64]
  %004_convolutional_conv_weights[FLOAT, 64x32x3x3]
  %006_convolutional_bn_scale[FLOAT, 128]
  %006_convolutional_bn_bias[FLOAT, 128]
  %006_convolutional_bn_mean[FLOAT, 128]
  %006_convolutional_bn_var[FLOAT, 128]
  %006_convolutional_conv_weights[FLOAT, 128x64x3x3]
  %007_convolutional_bn_scale[FLOAT, 64]
  %007_convolutional_bn_bias[FLOAT, 64]
  %007_convolutional_bn_mean[FLOAT, 64]
  %007_convolutional_bn_var[FLOAT, 64]
  %007_convolutional_conv_weights[FLOAT, 64x128x1x1]
  %008_convolutional_bn_scale[FLOAT, 128]
  %008_convolutional_bn_bias[FLOAT, 128]
  %008_convolutional_bn_mean[FLOAT, 128]
  %008_convolutional_bn_var[FLOAT, 128]
  %008_convolutional_conv_weights[FLOAT, 128x64x3x3]
  %010_convolutional_bn_scale[FLOAT, 64]
  %010_convolutional_bn_bias[FLOAT, 64]
  %010_convolutional_bn_mean[FLOAT, 64]
  %010_convolutional_bn_var[FLOAT, 64]
  %010_convolutional_conv_weights[FLOAT, 64x128x1x1]
  %011_convolutional_bn_scale[FLOAT, 128]
  %011_convolutional_bn_bias[FLOAT, 128]
  %011_convolutional_bn_mean[FLOAT, 128]
  %011_convolutional_bn_var[FLOAT, 128]
  %011_convolutional_conv_weights[FLOAT, 128x64x3x3]
  %013_convolutional_bn_scale[FLOAT, 256]
  %013_convolutional_bn_bias[FLOAT, 256]
  %013_convolutional_bn_mean[FLOAT, 256]
  %013_convolutional_bn_var[FLOAT, 256]
  %013_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %014_convolutional_bn_scale[FLOAT, 128]
  %014_convolutional_bn_bias[FLOAT, 128]
  %014_convolutional_bn_mean[FLOAT, 128]
  %014_convolutional_bn_var[FLOAT, 128]
  %014_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %015_convolutional_bn_scale[FLOAT, 256]
  %015_convolutional_bn_bias[FLOAT, 256]
  %015_convolutional_bn_mean[FLOAT, 256]
  %015_convolutional_bn_var[FLOAT, 256]
  %015_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %017_convolutional_bn_scale[FLOAT, 128]
  %017_convolutional_bn_bias[FLOAT, 128]
  %017_convolutional_bn_mean[FLOAT, 128]
  %017_convolutional_bn_var[FLOAT, 128]
  %017_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %018_convolutional_bn_scale[FLOAT, 256]
  %018_convolutional_bn_bias[FLOAT, 256]
  %018_convolutional_bn_mean[FLOAT, 256]
  %018_convolutional_bn_var[FLOAT, 256]
  %018_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %020_convolutional_bn_scale[FLOAT, 128]
  %020_convolutional_bn_bias[FLOAT, 128]
  %020_convolutional_bn_mean[FLOAT, 128]
  %020_convolutional_bn_var[FLOAT, 128]
  %020_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %021_convolutional_bn_scale[FLOAT, 256]
  %021_convolutional_bn_bias[FLOAT, 256]
  %021_convolutional_bn_mean[FLOAT, 256]
  %021_convolutional_bn_var[FLOAT, 256]
  %021_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %023_convolutional_bn_scale[FLOAT, 128]
  %023_convolutional_bn_bias[FLOAT, 128]
  %023_convolutional_bn_mean[FLOAT, 128]
  %023_convolutional_bn_var[FLOAT, 128]
  %023_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %024_convolutional_bn_scale[FLOAT, 256]
  %024_convolutional_bn_bias[FLOAT, 256]
  %024_convolutional_bn_mean[FLOAT, 256]
  %024_convolutional_bn_var[FLOAT, 256]
  %024_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %026_convolutional_bn_scale[FLOAT, 128]
  %026_convolutional_bn_bias[FLOAT, 128]
  %026_convolutional_bn_mean[FLOAT, 128]
  %026_convolutional_bn_var[FLOAT, 128]
  %026_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %027_convolutional_bn_scale[FLOAT, 256]
  %027_convolutional_bn_bias[FLOAT, 256]
  %027_convolutional_bn_mean[FLOAT, 256]
  %027_convolutional_bn_var[FLOAT, 256]
  %027_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %029_convolutional_bn_scale[FLOAT, 128]
  %029_convolutional_bn_bias[FLOAT, 128]
  %029_convolutional_bn_mean[FLOAT, 128]
  %029_convolutional_bn_var[FLOAT, 128]
  %029_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %030_convolutional_bn_scale[FLOAT, 256]
  %030_convolutional_bn_bias[FLOAT, 256]
  %030_convolutional_bn_mean[FLOAT, 256]
  %030_convolutional_bn_var[FLOAT, 256]
  %030_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %032_convolutional_bn_scale[FLOAT, 128]
  %032_convolutional_bn_bias[FLOAT, 128]
  %032_convolutional_bn_mean[FLOAT, 128]
  %032_convolutional_bn_var[FLOAT, 128]
  %032_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %033_convolutional_bn_scale[FLOAT, 256]
  %033_convolutional_bn_bias[FLOAT, 256]
  %033_convolutional_bn_mean[FLOAT, 256]
  %033_convolutional_bn_var[FLOAT, 256]
  %033_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %035_convolutional_bn_scale[FLOAT, 128]
  %035_convolutional_bn_bias[FLOAT, 128]
  %035_convolutional_bn_mean[FLOAT, 128]
  %035_convolutional_bn_var[FLOAT, 128]
  %035_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %036_convolutional_bn_scale[FLOAT, 256]
  %036_convolutional_bn_bias[FLOAT, 256]
  %036_convolutional_bn_mean[FLOAT, 256]
  %036_convolutional_bn_var[FLOAT, 256]
  %036_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %038_convolutional_bn_scale[FLOAT, 512]
  %038_convolutional_bn_bias[FLOAT, 512]
  %038_convolutional_bn_mean[FLOAT, 512]
  %038_convolutional_bn_var[FLOAT, 512]
  %038_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %039_convolutional_bn_scale[FLOAT, 256]
  %039_convolutional_bn_bias[FLOAT, 256]
  %039_convolutional_bn_mean[FLOAT, 256]
  %039_convolutional_bn_var[FLOAT, 256]
  %039_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %040_convolutional_bn_scale[FLOAT, 512]
  %040_convolutional_bn_bias[FLOAT, 512]
  %040_convolutional_bn_mean[FLOAT, 512]
  %040_convolutional_bn_var[FLOAT, 512]
  %040_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %042_convolutional_bn_scale[FLOAT, 256]
  %042_convolutional_bn_bias[FLOAT, 256]
  %042_convolutional_bn_mean[FLOAT, 256]
  %042_convolutional_bn_var[FLOAT, 256]
  %042_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %043_convolutional_bn_scale[FLOAT, 512]
  %043_convolutional_bn_bias[FLOAT, 512]
  %043_convolutional_bn_mean[FLOAT, 512]
  %043_convolutional_bn_var[FLOAT, 512]
  %043_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %045_convolutional_bn_scale[FLOAT, 256]
  %045_convolutional_bn_bias[FLOAT, 256]
  %045_convolutional_bn_mean[FLOAT, 256]
  %045_convolutional_bn_var[FLOAT, 256]
  %045_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %046_convolutional_bn_scale[FLOAT, 512]
  %046_convolutional_bn_bias[FLOAT, 512]
  %046_convolutional_bn_mean[FLOAT, 512]
  %046_convolutional_bn_var[FLOAT, 512]
  %046_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %048_convolutional_bn_scale[FLOAT, 256]
  %048_convolutional_bn_bias[FLOAT, 256]
  %048_convolutional_bn_mean[FLOAT, 256]
  %048_convolutional_bn_var[FLOAT, 256]
  %048_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %049_convolutional_bn_scale[FLOAT, 512]
  %049_convolutional_bn_bias[FLOAT, 512]
  %049_convolutional_bn_mean[FLOAT, 512]
  %049_convolutional_bn_var[FLOAT, 512]
  %049_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %051_convolutional_bn_scale[FLOAT, 256]
  %051_convolutional_bn_bias[FLOAT, 256]
  %051_convolutional_bn_mean[FLOAT, 256]
  %051_convolutional_bn_var[FLOAT, 256]
  %051_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %052_convolutional_bn_scale[FLOAT, 512]
  %052_convolutional_bn_bias[FLOAT, 512]
  %052_convolutional_bn_mean[FLOAT, 512]
  %052_convolutional_bn_var[FLOAT, 512]
  %052_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %054_convolutional_bn_scale[FLOAT, 256]
  %054_convolutional_bn_bias[FLOAT, 256]
  %054_convolutional_bn_mean[FLOAT, 256]
  %054_convolutional_bn_var[FLOAT, 256]
  %054_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %055_convolutional_bn_scale[FLOAT, 512]
  %055_convolutional_bn_bias[FLOAT, 512]
  %055_convolutional_bn_mean[FLOAT, 512]
  %055_convolutional_bn_var[FLOAT, 512]
  %055_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %057_convolutional_bn_scale[FLOAT, 256]
  %057_convolutional_bn_bias[FLOAT, 256]
  %057_convolutional_bn_mean[FLOAT, 256]
  %057_convolutional_bn_var[FLOAT, 256]
  %057_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %058_convolutional_bn_scale[FLOAT, 512]
  %058_convolutional_bn_bias[FLOAT, 512]
  %058_convolutional_bn_mean[FLOAT, 512]
  %058_convolutional_bn_var[FLOAT, 512]
  %058_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %060_convolutional_bn_scale[FLOAT, 256]
  %060_convolutional_bn_bias[FLOAT, 256]
  %060_convolutional_bn_mean[FLOAT, 256]
  %060_convolutional_bn_var[FLOAT, 256]
  %060_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %061_convolutional_bn_scale[FLOAT, 512]
  %061_convolutional_bn_bias[FLOAT, 512]
  %061_convolutional_bn_mean[FLOAT, 512]
  %061_convolutional_bn_var[FLOAT, 512]
  %061_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %063_convolutional_bn_scale[FLOAT, 1024]
  %063_convolutional_bn_bias[FLOAT, 1024]
  %063_convolutional_bn_mean[FLOAT, 1024]
  %063_convolutional_bn_var[FLOAT, 1024]
  %063_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %064_convolutional_bn_scale[FLOAT, 512]
  %064_convolutional_bn_bias[FLOAT, 512]
  %064_convolutional_bn_mean[FLOAT, 512]
  %064_convolutional_bn_var[FLOAT, 512]
  %064_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %065_convolutional_bn_scale[FLOAT, 1024]
  %065_convolutional_bn_bias[FLOAT, 1024]
  %065_convolutional_bn_mean[FLOAT, 1024]
  %065_convolutional_bn_var[FLOAT, 1024]
  %065_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %067_convolutional_bn_scale[FLOAT, 512]
  %067_convolutional_bn_bias[FLOAT, 512]
  %067_convolutional_bn_mean[FLOAT, 512]
  %067_convolutional_bn_var[FLOAT, 512]
  %067_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %068_convolutional_bn_scale[FLOAT, 1024]
  %068_convolutional_bn_bias[FLOAT, 1024]
  %068_convolutional_bn_mean[FLOAT, 1024]
  %068_convolutional_bn_var[FLOAT, 1024]
  %068_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %070_convolutional_bn_scale[FLOAT, 512]
  %070_convolutional_bn_bias[FLOAT, 512]
  %070_convolutional_bn_mean[FLOAT, 512]
  %070_convolutional_bn_var[FLOAT, 512]
  %070_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %071_convolutional_bn_scale[FLOAT, 1024]
  %071_convolutional_bn_bias[FLOAT, 1024]
  %071_convolutional_bn_mean[FLOAT, 1024]
  %071_convolutional_bn_var[FLOAT, 1024]
  %071_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %073_convolutional_bn_scale[FLOAT, 512]
  %073_convolutional_bn_bias[FLOAT, 512]
  %073_convolutional_bn_mean[FLOAT, 512]
  %073_convolutional_bn_var[FLOAT, 512]
  %073_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %074_convolutional_bn_scale[FLOAT, 1024]
  %074_convolutional_bn_bias[FLOAT, 1024]
  %074_convolutional_bn_mean[FLOAT, 1024]
  %074_convolutional_bn_var[FLOAT, 1024]
  %074_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %076_convolutional_bn_scale[FLOAT, 512]
  %076_convolutional_bn_bias[FLOAT, 512]
  %076_convolutional_bn_mean[FLOAT, 512]
  %076_convolutional_bn_var[FLOAT, 512]
  %076_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %077_convolutional_bn_scale[FLOAT, 1024]
  %077_convolutional_bn_bias[FLOAT, 1024]
  %077_convolutional_bn_mean[FLOAT, 1024]
  %077_convolutional_bn_var[FLOAT, 1024]
  %077_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %078_convolutional_bn_scale[FLOAT, 512]
  %078_convolutional_bn_bias[FLOAT, 512]
  %078_convolutional_bn_mean[FLOAT, 512]
  %078_convolutional_bn_var[FLOAT, 512]
  %078_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %079_convolutional_bn_scale[FLOAT, 1024]
  %079_convolutional_bn_bias[FLOAT, 1024]
  %079_convolutional_bn_mean[FLOAT, 1024]
  %079_convolutional_bn_var[FLOAT, 1024]
  %079_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %080_convolutional_bn_scale[FLOAT, 512]
  %080_convolutional_bn_bias[FLOAT, 512]
  %080_convolutional_bn_mean[FLOAT, 512]
  %080_convolutional_bn_var[FLOAT, 512]
  %080_convolutional_conv_weights[FLOAT, 512x1024x1x1]
  %081_convolutional_bn_scale[FLOAT, 1024]
  %081_convolutional_bn_bias[FLOAT, 1024]
  %081_convolutional_bn_mean[FLOAT, 1024]
  %081_convolutional_bn_var[FLOAT, 1024]
  %081_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %082_convolutional_conv_bias[FLOAT, 255]
  %082_convolutional_conv_weights[FLOAT, 255x1024x1x1]
  %085_convolutional_bn_scale[FLOAT, 256]
  %085_convolutional_bn_bias[FLOAT, 256]
  %085_convolutional_bn_mean[FLOAT, 256]
  %085_convolutional_bn_var[FLOAT, 256]
  %085_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %086_upsample_scale[FLOAT, 4]
  %088_convolutional_bn_scale[FLOAT, 256]
  %088_convolutional_bn_bias[FLOAT, 256]
  %088_convolutional_bn_mean[FLOAT, 256]
  %088_convolutional_bn_var[FLOAT, 256]
  %088_convolutional_conv_weights[FLOAT, 256x768x1x1]
  %089_convolutional_bn_scale[FLOAT, 512]
  %089_convolutional_bn_bias[FLOAT, 512]
  %089_convolutional_bn_mean[FLOAT, 512]
  %089_convolutional_bn_var[FLOAT, 512]
  %089_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %090_convolutional_bn_scale[FLOAT, 256]
  %090_convolutional_bn_bias[FLOAT, 256]
  %090_convolutional_bn_mean[FLOAT, 256]
  %090_convolutional_bn_var[FLOAT, 256]
  %090_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %091_convolutional_bn_scale[FLOAT, 512]
  %091_convolutional_bn_bias[FLOAT, 512]
  %091_convolutional_bn_mean[FLOAT, 512]
  %091_convolutional_bn_var[FLOAT, 512]
  %091_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %092_convolutional_bn_scale[FLOAT, 256]
  %092_convolutional_bn_bias[FLOAT, 256]
  %092_convolutional_bn_mean[FLOAT, 256]
  %092_convolutional_bn_var[FLOAT, 256]
  %092_convolutional_conv_weights[FLOAT, 256x512x1x1]
  %093_convolutional_bn_scale[FLOAT, 512]
  %093_convolutional_bn_bias[FLOAT, 512]
  %093_convolutional_bn_mean[FLOAT, 512]
  %093_convolutional_bn_var[FLOAT, 512]
  %093_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %094_convolutional_conv_bias[FLOAT, 255]
  %094_convolutional_conv_weights[FLOAT, 255x512x1x1]
  %097_convolutional_bn_scale[FLOAT, 128]
  %097_convolutional_bn_bias[FLOAT, 128]
  %097_convolutional_bn_mean[FLOAT, 128]
  %097_convolutional_bn_var[FLOAT, 128]
  %097_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %098_upsample_scale[FLOAT, 4]
  %100_convolutional_bn_scale[FLOAT, 128]
  %100_convolutional_bn_bias[FLOAT, 128]
  %100_convolutional_bn_mean[FLOAT, 128]
  %100_convolutional_bn_var[FLOAT, 128]
  %100_convolutional_conv_weights[FLOAT, 128x384x1x1]
  %101_convolutional_bn_scale[FLOAT, 256]
  %101_convolutional_bn_bias[FLOAT, 256]
  %101_convolutional_bn_mean[FLOAT, 256]
  %101_convolutional_bn_var[FLOAT, 256]
  %101_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %102_convolutional_bn_scale[FLOAT, 128]
  %102_convolutional_bn_bias[FLOAT, 128]
  %102_convolutional_bn_mean[FLOAT, 128]
  %102_convolutional_bn_var[FLOAT, 128]
  %102_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %103_convolutional_bn_scale[FLOAT, 256]
  %103_convolutional_bn_bias[FLOAT, 256]
  %103_convolutional_bn_mean[FLOAT, 256]
  %103_convolutional_bn_var[FLOAT, 256]
  %103_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %104_convolutional_bn_scale[FLOAT, 128]
  %104_convolutional_bn_bias[FLOAT, 128]
  %104_convolutional_bn_mean[FLOAT, 128]
  %104_convolutional_bn_var[FLOAT, 128]
  %104_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %105_convolutional_bn_scale[FLOAT, 256]
  %105_convolutional_bn_bias[FLOAT, 256]
  %105_convolutional_bn_mean[FLOAT, 256]
  %105_convolutional_bn_var[FLOAT, 256]
  %105_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %106_convolutional_conv_bias[FLOAT, 255]
  %106_convolutional_conv_weights[FLOAT, 255x256x1x1]
) {
  %001_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%000_net, %001_convolutional_conv_weights)
  %001_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%001_convolutional, %001_convolutional_bn_scale, %001_convolutional_bn_bias, %001_convolutional_bn_mean, %001_convolutional_bn_var)
  %001_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%001_convolutional_bn)
  %002_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [2, 2]](%001_convolutional_lrelu, %002_convolutional_conv_weights)
  %002_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%002_convolutional, %002_convolutional_bn_scale, %002_convolutional_bn_bias, %002_convolutional_bn_mean, %002_convolutional_bn_var)
  %002_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%002_convolutional_bn)
  %003_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%002_convolutional_lrelu, %003_convolutional_conv_weights)
  %003_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%003_convolutional, %003_convolutional_bn_scale, %003_convolutional_bn_bias, %003_convolutional_bn_mean, %003_convolutional_bn_var)
  %003_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%003_convolutional_bn)
  %004_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%003_convolutional_lrelu, %004_convolutional_conv_weights)
  %004_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%004_convolutional, %004_convolutional_bn_scale, %004_convolutional_bn_bias, %004_convolutional_bn_mean, %004_convolutional_bn_var)
  %004_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%004_convolutional_bn)
  %005_shortcut = Add(%004_convolutional_lrelu, %002_convolutional_lrelu)
  %006_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [2, 2]](%005_shortcut, %006_convolutional_conv_weights)
  %006_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%006_convolutional, %006_convolutional_bn_scale, %006_convolutional_bn_bias, %006_convolutional_bn_mean, %006_convolutional_bn_var)
  %006_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%006_convolutional_bn)
  %007_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%006_convolutional_lrelu, %007_convolutional_conv_weights)
  %007_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%007_convolutional, %007_convolutional_bn_scale, %007_convolutional_bn_bias, %007_convolutional_bn_mean, %007_convolutional_bn_var)
  %007_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%007_convolutional_bn)
  %008_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%007_convolutional_lrelu, %008_convolutional_conv_weights)
  %008_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%008_convolutional, %008_convolutional_bn_scale, %008_convolutional_bn_bias, %008_convolutional_bn_mean, %008_convolutional_bn_var)
  %008_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%008_convolutional_bn)
  %009_shortcut = Add(%008_convolutional_lrelu, %006_convolutional_lrelu)
  %010_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%009_shortcut, %010_convolutional_conv_weights)
  %010_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%010_convolutional, %010_convolutional_bn_scale, %010_convolutional_bn_bias, %010_convolutional_bn_mean, %010_convolutional_bn_var)
  %010_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%010_convolutional_bn)
  %011_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%010_convolutional_lrelu, %011_convolutional_conv_weights)
  %011_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%011_convolutional, %011_convolutional_bn_scale, %011_convolutional_bn_bias, %011_convolutional_bn_mean, %011_convolutional_bn_var)
  %011_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%011_convolutional_bn)
  %012_shortcut = Add(%011_convolutional_lrelu, %009_shortcut)
  %013_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [2, 2]](%012_shortcut, %013_convolutional_conv_weights)
  %013_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%013_convolutional, %013_convolutional_bn_scale, %013_convolutional_bn_bias, %013_convolutional_bn_mean, %013_convolutional_bn_var)
  %013_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%013_convolutional_bn)
  %014_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%013_convolutional_lrelu, %014_convolutional_conv_weights)
  %014_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%014_convolutional, %014_convolutional_bn_scale, %014_convolutional_bn_bias, %014_convolutional_bn_mean, %014_convolutional_bn_var)
  %014_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%014_convolutional_bn)
  %015_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%014_convolutional_lrelu, %015_convolutional_conv_weights)
  %015_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%015_convolutional, %015_convolutional_bn_scale, %015_convolutional_bn_bias, %015_convolutional_bn_mean, %015_convolutional_bn_var)
  %015_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%015_convolutional_bn)
  %016_shortcut = Add(%015_convolutional_lrelu, %013_convolutional_lrelu)
  %017_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%016_shortcut, %017_convolutional_conv_weights)
  %017_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%017_convolutional, %017_convolutional_bn_scale, %017_convolutional_bn_bias, %017_convolutional_bn_mean, %017_convolutional_bn_var)
  %017_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%017_convolutional_bn)
  %018_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%017_convolutional_lrelu, %018_convolutional_conv_weights)
  %018_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%018_convolutional, %018_convolutional_bn_scale, %018_convolutional_bn_bias, %018_convolutional_bn_mean, %018_convolutional_bn_var)
  %018_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%018_convolutional_bn)
  %019_shortcut = Add(%018_convolutional_lrelu, %016_shortcut)
  %020_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%019_shortcut, %020_convolutional_conv_weights)
  %020_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%020_convolutional, %020_convolutional_bn_scale, %020_convolutional_bn_bias, %020_convolutional_bn_mean, %020_convolutional_bn_var)
  %020_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%020_convolutional_bn)
  %021_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%020_convolutional_lrelu, %021_convolutional_conv_weights)
  %021_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%021_convolutional, %021_convolutional_bn_scale, %021_convolutional_bn_bias, %021_convolutional_bn_mean, %021_convolutional_bn_var)
  %021_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%021_convolutional_bn)
  %022_shortcut = Add(%021_convolutional_lrelu, %019_shortcut)
  %023_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%022_shortcut, %023_convolutional_conv_weights)
  %023_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%023_convolutional, %023_convolutional_bn_scale, %023_convolutional_bn_bias, %023_convolutional_bn_mean, %023_convolutional_bn_var)
  %023_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%023_convolutional_bn)
  %024_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%023_convolutional_lrelu, %024_convolutional_conv_weights)
  %024_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%024_convolutional, %024_convolutional_bn_scale, %024_convolutional_bn_bias, %024_convolutional_bn_mean, %024_convolutional_bn_var)
  %024_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%024_convolutional_bn)
  %025_shortcut = Add(%024_convolutional_lrelu, %022_shortcut)
  %026_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%025_shortcut, %026_convolutional_conv_weights)
  %026_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%026_convolutional, %026_convolutional_bn_scale, %026_convolutional_bn_bias, %026_convolutional_bn_mean, %026_convolutional_bn_var)
  %026_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%026_convolutional_bn)
  %027_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%026_convolutional_lrelu, %027_convolutional_conv_weights)
  %027_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%027_convolutional, %027_convolutional_bn_scale, %027_convolutional_bn_bias, %027_convolutional_bn_mean, %027_convolutional_bn_var)
  %027_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%027_convolutional_bn)
  %028_shortcut = Add(%027_convolutional_lrelu, %025_shortcut)
  %029_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%028_shortcut, %029_convolutional_conv_weights)
  %029_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%029_convolutional, %029_convolutional_bn_scale, %029_convolutional_bn_bias, %029_convolutional_bn_mean, %029_convolutional_bn_var)
  %029_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%029_convolutional_bn)
  %030_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%029_convolutional_lrelu, %030_convolutional_conv_weights)
  %030_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%030_convolutional, %030_convolutional_bn_scale, %030_convolutional_bn_bias, %030_convolutional_bn_mean, %030_convolutional_bn_var)
  %030_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%030_convolutional_bn)
  %031_shortcut = Add(%030_convolutional_lrelu, %028_shortcut)
  %032_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%031_shortcut, %032_convolutional_conv_weights)
  %032_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%032_convolutional, %032_convolutional_bn_scale, %032_convolutional_bn_bias, %032_convolutional_bn_mean, %032_convolutional_bn_var)
  %032_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%032_convolutional_bn)
  %033_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%032_convolutional_lrelu, %033_convolutional_conv_weights)
  %033_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%033_convolutional, %033_convolutional_bn_scale, %033_convolutional_bn_bias, %033_convolutional_bn_mean, %033_convolutional_bn_var)
  %033_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%033_convolutional_bn)
  %034_shortcut = Add(%033_convolutional_lrelu, %031_shortcut)
  %035_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%034_shortcut, %035_convolutional_conv_weights)
  %035_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%035_convolutional, %035_convolutional_bn_scale, %035_convolutional_bn_bias, %035_convolutional_bn_mean, %035_convolutional_bn_var)
  %035_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%035_convolutional_bn)
  %036_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%035_convolutional_lrelu, %036_convolutional_conv_weights)
  %036_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%036_convolutional, %036_convolutional_bn_scale, %036_convolutional_bn_bias, %036_convolutional_bn_mean, %036_convolutional_bn_var)
  %036_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%036_convolutional_bn)
  %037_shortcut = Add(%036_convolutional_lrelu, %034_shortcut)
  %038_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [2, 2]](%037_shortcut, %038_convolutional_conv_weights)
  %038_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%038_convolutional, %038_convolutional_bn_scale, %038_convolutional_bn_bias, %038_convolutional_bn_mean, %038_convolutional_bn_var)
  %038_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%038_convolutional_bn)
  %039_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%038_convolutional_lrelu, %039_convolutional_conv_weights)
  %039_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%039_convolutional, %039_convolutional_bn_scale, %039_convolutional_bn_bias, %039_convolutional_bn_mean, %039_convolutional_bn_var)
  %039_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%039_convolutional_bn)
  %040_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%039_convolutional_lrelu, %040_convolutional_conv_weights)
  %040_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%040_convolutional, %040_convolutional_bn_scale, %040_convolutional_bn_bias, %040_convolutional_bn_mean, %040_convolutional_bn_var)
  %040_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%040_convolutional_bn)
  %041_shortcut = Add(%040_convolutional_lrelu, %038_convolutional_lrelu)
  %042_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%041_shortcut, %042_convolutional_conv_weights)
  %042_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%042_convolutional, %042_convolutional_bn_scale, %042_convolutional_bn_bias, %042_convolutional_bn_mean, %042_convolutional_bn_var)
  %042_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%042_convolutional_bn)
  %043_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%042_convolutional_lrelu, %043_convolutional_conv_weights)
  %043_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%043_convolutional, %043_convolutional_bn_scale, %043_convolutional_bn_bias, %043_convolutional_bn_mean, %043_convolutional_bn_var)
  %043_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%043_convolutional_bn)
  %044_shortcut = Add(%043_convolutional_lrelu, %041_shortcut)
  %045_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%044_shortcut, %045_convolutional_conv_weights)
  %045_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%045_convolutional, %045_convolutional_bn_scale, %045_convolutional_bn_bias, %045_convolutional_bn_mean, %045_convolutional_bn_var)
  %045_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%045_convolutional_bn)
  %046_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%045_convolutional_lrelu, %046_convolutional_conv_weights)
  %046_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%046_convolutional, %046_convolutional_bn_scale, %046_convolutional_bn_bias, %046_convolutional_bn_mean, %046_convolutional_bn_var)
  %046_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%046_convolutional_bn)
  %047_shortcut = Add(%046_convolutional_lrelu, %044_shortcut)
  %048_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%047_shortcut, %048_convolutional_conv_weights)
  %048_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%048_convolutional, %048_convolutional_bn_scale, %048_convolutional_bn_bias, %048_convolutional_bn_mean, %048_convolutional_bn_var)
  %048_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%048_convolutional_bn)
  %049_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%048_convolutional_lrelu, %049_convolutional_conv_weights)
  %049_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%049_convolutional, %049_convolutional_bn_scale, %049_convolutional_bn_bias, %049_convolutional_bn_mean, %049_convolutional_bn_var)
  %049_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%049_convolutional_bn)
  %050_shortcut = Add(%049_convolutional_lrelu, %047_shortcut)
  %051_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%050_shortcut, %051_convolutional_conv_weights)
  %051_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%051_convolutional, %051_convolutional_bn_scale, %051_convolutional_bn_bias, %051_convolutional_bn_mean, %051_convolutional_bn_var)
  %051_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%051_convolutional_bn)
  %052_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%051_convolutional_lrelu, %052_convolutional_conv_weights)
  %052_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%052_convolutional, %052_convolutional_bn_scale, %052_convolutional_bn_bias, %052_convolutional_bn_mean, %052_convolutional_bn_var)
  %052_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%052_convolutional_bn)
  %053_shortcut = Add(%052_convolutional_lrelu, %050_shortcut)
  %054_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%053_shortcut, %054_convolutional_conv_weights)
  %054_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%054_convolutional, %054_convolutional_bn_scale, %054_convolutional_bn_bias, %054_convolutional_bn_mean, %054_convolutional_bn_var)
  %054_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%054_convolutional_bn)
  %055_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%054_convolutional_lrelu, %055_convolutional_conv_weights)
  %055_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%055_convolutional, %055_convolutional_bn_scale, %055_convolutional_bn_bias, %055_convolutional_bn_mean, %055_convolutional_bn_var)
  %055_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%055_convolutional_bn)
  %056_shortcut = Add(%055_convolutional_lrelu, %053_shortcut)
  %057_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%056_shortcut, %057_convolutional_conv_weights)
  %057_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%057_convolutional, %057_convolutional_bn_scale, %057_convolutional_bn_bias, %057_convolutional_bn_mean, %057_convolutional_bn_var)
  %057_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%057_convolutional_bn)
  %058_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%057_convolutional_lrelu, %058_convolutional_conv_weights)
  %058_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%058_convolutional, %058_convolutional_bn_scale, %058_convolutional_bn_bias, %058_convolutional_bn_mean, %058_convolutional_bn_var)
  %058_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%058_convolutional_bn)
  %059_shortcut = Add(%058_convolutional_lrelu, %056_shortcut)
  %060_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%059_shortcut, %060_convolutional_conv_weights)
  %060_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%060_convolutional, %060_convolutional_bn_scale, %060_convolutional_bn_bias, %060_convolutional_bn_mean, %060_convolutional_bn_var)
  %060_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%060_convolutional_bn)
  %061_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%060_convolutional_lrelu, %061_convolutional_conv_weights)
  %061_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%061_convolutional, %061_convolutional_bn_scale, %061_convolutional_bn_bias, %061_convolutional_bn_mean, %061_convolutional_bn_var)
  %061_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%061_convolutional_bn)
  %062_shortcut = Add(%061_convolutional_lrelu, %059_shortcut)
  %063_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [2, 2]](%062_shortcut, %063_convolutional_conv_weights)
  %063_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%063_convolutional, %063_convolutional_bn_scale, %063_convolutional_bn_bias, %063_convolutional_bn_mean, %063_convolutional_bn_var)
  %063_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%063_convolutional_bn)
  %064_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%063_convolutional_lrelu, %064_convolutional_conv_weights)
  %064_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%064_convolutional, %064_convolutional_bn_scale, %064_convolutional_bn_bias, %064_convolutional_bn_mean, %064_convolutional_bn_var)
  %064_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%064_convolutional_bn)
  %065_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%064_convolutional_lrelu, %065_convolutional_conv_weights)
  %065_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%065_convolutional, %065_convolutional_bn_scale, %065_convolutional_bn_bias, %065_convolutional_bn_mean, %065_convolutional_bn_var)
  %065_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%065_convolutional_bn)
  %066_shortcut = Add(%065_convolutional_lrelu, %063_convolutional_lrelu)
  %067_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%066_shortcut, %067_convolutional_conv_weights)
  %067_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%067_convolutional, %067_convolutional_bn_scale, %067_convolutional_bn_bias, %067_convolutional_bn_mean, %067_convolutional_bn_var)
  %067_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%067_convolutional_bn)
  %068_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%067_convolutional_lrelu, %068_convolutional_conv_weights)
  %068_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%068_convolutional, %068_convolutional_bn_scale, %068_convolutional_bn_bias, %068_convolutional_bn_mean, %068_convolutional_bn_var)
  %068_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%068_convolutional_bn)
  %069_shortcut = Add(%068_convolutional_lrelu, %066_shortcut)
  %070_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%069_shortcut, %070_convolutional_conv_weights)
  %070_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%070_convolutional, %070_convolutional_bn_scale, %070_convolutional_bn_bias, %070_convolutional_bn_mean, %070_convolutional_bn_var)
  %070_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%070_convolutional_bn)
  %071_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%070_convolutional_lrelu, %071_convolutional_conv_weights)
  %071_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%071_convolutional, %071_convolutional_bn_scale, %071_convolutional_bn_bias, %071_convolutional_bn_mean, %071_convolutional_bn_var)
  %071_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%071_convolutional_bn)
  %072_shortcut = Add(%071_convolutional_lrelu, %069_shortcut)
  %073_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%072_shortcut, %073_convolutional_conv_weights)
  %073_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%073_convolutional, %073_convolutional_bn_scale, %073_convolutional_bn_bias, %073_convolutional_bn_mean, %073_convolutional_bn_var)
  %073_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%073_convolutional_bn)
  %074_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%073_convolutional_lrelu, %074_convolutional_conv_weights)
  %074_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%074_convolutional, %074_convolutional_bn_scale, %074_convolutional_bn_bias, %074_convolutional_bn_mean, %074_convolutional_bn_var)
  %074_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%074_convolutional_bn)
  %075_shortcut = Add(%074_convolutional_lrelu, %072_shortcut)
  %076_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%075_shortcut, %076_convolutional_conv_weights)
  %076_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%076_convolutional, %076_convolutional_bn_scale, %076_convolutional_bn_bias, %076_convolutional_bn_mean, %076_convolutional_bn_var)
  %076_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%076_convolutional_bn)
  %077_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%076_convolutional_lrelu, %077_convolutional_conv_weights)
  %077_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%077_convolutional, %077_convolutional_bn_scale, %077_convolutional_bn_bias, %077_convolutional_bn_mean, %077_convolutional_bn_var)
  %077_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%077_convolutional_bn)
  %078_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%077_convolutional_lrelu, %078_convolutional_conv_weights)
  %078_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%078_convolutional, %078_convolutional_bn_scale, %078_convolutional_bn_bias, %078_convolutional_bn_mean, %078_convolutional_bn_var)
  %078_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%078_convolutional_bn)
  %079_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%078_convolutional_lrelu, %079_convolutional_conv_weights)
  %079_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%079_convolutional, %079_convolutional_bn_scale, %079_convolutional_bn_bias, %079_convolutional_bn_mean, %079_convolutional_bn_var)
  %079_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%079_convolutional_bn)
  %080_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%079_convolutional_lrelu, %080_convolutional_conv_weights)
  %080_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%080_convolutional, %080_convolutional_bn_scale, %080_convolutional_bn_bias, %080_convolutional_bn_mean, %080_convolutional_bn_var)
  %080_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%080_convolutional_bn)
  %081_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%080_convolutional_lrelu, %081_convolutional_conv_weights)
  %081_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%081_convolutional, %081_convolutional_bn_scale, %081_convolutional_bn_bias, %081_convolutional_bn_mean, %081_convolutional_bn_var)
  %081_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%081_convolutional_bn)
  %082_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%081_convolutional_lrelu, %082_convolutional_conv_weights, %082_convolutional_conv_bias)
  %085_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%080_convolutional_lrelu, %085_convolutional_conv_weights)
  %085_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%085_convolutional, %085_convolutional_bn_scale, %085_convolutional_bn_bias, %085_convolutional_bn_mean, %085_convolutional_bn_var)
  %085_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%085_convolutional_bn)
  %086_upsample = Resize[mode = 'nearest'](%085_convolutional_lrelu, %086_upsample_scale)
  %087_route = Concat[axis = 1](%086_upsample, %062_shortcut)
  %088_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%087_route, %088_convolutional_conv_weights)
  %088_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%088_convolutional, %088_convolutional_bn_scale, %088_convolutional_bn_bias, %088_convolutional_bn_mean, %088_convolutional_bn_var)
  %088_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%088_convolutional_bn)
  %089_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%088_convolutional_lrelu, %089_convolutional_conv_weights)
  %089_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%089_convolutional, %089_convolutional_bn_scale, %089_convolutional_bn_bias, %089_convolutional_bn_mean, %089_convolutional_bn_var)
  %089_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%089_convolutional_bn)
  %090_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%089_convolutional_lrelu, %090_convolutional_conv_weights)
  %090_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%090_convolutional, %090_convolutional_bn_scale, %090_convolutional_bn_bias, %090_convolutional_bn_mean, %090_convolutional_bn_var)
  %090_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%090_convolutional_bn)
  %091_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%090_convolutional_lrelu, %091_convolutional_conv_weights)
  %091_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%091_convolutional, %091_convolutional_bn_scale, %091_convolutional_bn_bias, %091_convolutional_bn_mean, %091_convolutional_bn_var)
  %091_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%091_convolutional_bn)
  %092_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%091_convolutional_lrelu, %092_convolutional_conv_weights)
  %092_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%092_convolutional, %092_convolutional_bn_scale, %092_convolutional_bn_bias, %092_convolutional_bn_mean, %092_convolutional_bn_var)
  %092_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%092_convolutional_bn)
  %093_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%092_convolutional_lrelu, %093_convolutional_conv_weights)
  %093_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%093_convolutional, %093_convolutional_bn_scale, %093_convolutional_bn_bias, %093_convolutional_bn_mean, %093_convolutional_bn_var)
  %093_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%093_convolutional_bn)
  %094_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%093_convolutional_lrelu, %094_convolutional_conv_weights, %094_convolutional_conv_bias)
  %097_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%092_convolutional_lrelu, %097_convolutional_conv_weights)
  %097_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%097_convolutional, %097_convolutional_bn_scale, %097_convolutional_bn_bias, %097_convolutional_bn_mean, %097_convolutional_bn_var)
  %097_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%097_convolutional_bn)
  %098_upsample = Resize[mode = 'nearest'](%097_convolutional_lrelu, %098_upsample_scale)
  %099_route = Concat[axis = 1](%098_upsample, %037_shortcut)
  %100_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%099_route, %100_convolutional_conv_weights)
  %100_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%100_convolutional, %100_convolutional_bn_scale, %100_convolutional_bn_bias, %100_convolutional_bn_mean, %100_convolutional_bn_var)
  %100_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%100_convolutional_bn)
  %101_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%100_convolutional_lrelu, %101_convolutional_conv_weights)
  %101_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%101_convolutional, %101_convolutional_bn_scale, %101_convolutional_bn_bias, %101_convolutional_bn_mean, %101_convolutional_bn_var)
  %101_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%101_convolutional_bn)
  %102_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%101_convolutional_lrelu, %102_convolutional_conv_weights)
  %102_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%102_convolutional, %102_convolutional_bn_scale, %102_convolutional_bn_bias, %102_convolutional_bn_mean, %102_convolutional_bn_var)
  %102_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%102_convolutional_bn)
  %103_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%102_convolutional_lrelu, %103_convolutional_conv_weights)
  %103_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%103_convolutional, %103_convolutional_bn_scale, %103_convolutional_bn_bias, %103_convolutional_bn_mean, %103_convolutional_bn_var)
  %103_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%103_convolutional_bn)
  %104_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%103_convolutional_lrelu, %104_convolutional_conv_weights)
  %104_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%104_convolutional, %104_convolutional_bn_scale, %104_convolutional_bn_bias, %104_convolutional_bn_mean, %104_convolutional_bn_var)
  %104_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%104_convolutional_bn)
  %105_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%104_convolutional_lrelu, %105_convolutional_conv_weights)
  %105_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%105_convolutional, %105_convolutional_bn_scale, %105_convolutional_bn_bias, %105_convolutional_bn_mean, %105_convolutional_bn_var)
  %105_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%105_convolutional_bn)
  %106_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%105_convolutional_lrelu, %106_convolutional_conv_weights, %106_convolutional_conv_bias)
  return %082_convolutional, %094_convolutional, %106_convolutional
}