import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 2
    backbone        = 'vgg'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes = num_classes, backbone = backbone).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    # flops * 2 because profile does not include convolution as two operations
    # Some papers refer to convolution as multiplication and addition operations. I'm multiplying by 2
    # Some papers only consider the number of operations of multiplication and ignore addition. I'm not multiplying by 2
    # Multiply by 2 for this code. Refer to YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
