import torch
import numpy as np

def kernel2marix(kernel, in_row, in_col, stride):
    k_row, k_col = kernel.shape
    slide_cnt_v = (in_row - k_row) // stride + 1 #纵向滑动次数
    slide_cnt_h = (in_col - k_col) // stride + 1 #横向滑动次数
    matrix = []
    for y in range(slide_cnt_v):
        offy = y * stride
        for x in range(0, slide_cnt_h):
            offx = x * stride
            k = np.pad(kernel, [[offy, in_row - k_row - offy],
                                [offx, in_col - k_col - offx]], mode='constant')
            # print(k)
            k = k.flatten()
            #print(k.shape)
            matrix.append(k)
    matrix = np.array(matrix).T
    #print(matrix.shape)
    #print(matrix)
    return matrix

def run_conv(kernel_size=3, stride=1, padding=0):
    conv = torch.nn.Conv2d(in_channels=1,
                        out_channels=1,
                        kernel_size=[kernel_size, kernel_size],
                        stride=[stride, stride],
                        padding=[padding, padding],
                        bias=False)
    input_size = 5
    inputs = np.arange(1, 26).reshape(1, 1, 5, 5).astype(np.float32)

    inputs_tensor = torch.from_numpy(inputs)
    outputs = conv(inputs_tensor)
    print('pytorch conv:\n', outputs.squeeze().detach().numpy())

    inputs = inputs[0, 0, :, :]
    inputs = np.pad(inputs, [[padding, padding], [padding, padding]], mode='constant')
    inputs = inputs.flatten().reshape(1, -1)
    matrix = kernel2marix(conv.state_dict()['weight'].squeeze().detach().numpy(),
                            input_size + padding * 2, input_size + padding * 2, stride)
    outputs = np.matmul(inputs, matrix).reshape(outputs.squeeze().shape)
    print('test conv   :\n', outputs)

def run_conv_trans(kernel_size=3, stride=1, padding=0, output_padding=0):
    conv_trans = torch.nn.ConvTranspose2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=[kernel_size, kernel_size],
                                        stride=[stride, stride],
                                        padding=[padding, padding],
                                        output_padding=[output_padding, output_padding],
                                        bias=False)

    input_size = 3
    inputs = np.arange(1, 10).reshape(1, 1, input_size, input_size).astype(np.float32)
    
    inputs_tensor = torch.from_numpy(inputs)
    outputs = conv_trans(inputs_tensor)
    print('pytorch conv transpose:\n', outputs.squeeze().detach().numpy())

    inputs = inputs.flatten().reshape(1, -1)
    size = (input_size - 1) * stride + kernel_size + output_padding
    matrix = kernel2marix(conv_trans.state_dict()['weight'].squeeze().detach().numpy(),
                        size, size, stride).T
    outputs = np.matmul(inputs, matrix).reshape(size, size)
    outputs = outputs[padding:size-padding, padding:size-padding]
    print('test conv transpose   :\n', outputs)

if __name__ == '__main__':
    run_conv(stride=2, padding=1)
    run_conv_trans(stride=2, padding=2, output_padding=1)
