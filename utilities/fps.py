import os.path
import time
import torch
import torch.nn.functional as F
import sys
import numpy as np
import onnx
import onnxruntime
sys.path.append(os.path.abspath('..'))
from models.net_factory import get_model
from torchsummaryX import summary
from models.total_supvised.fastsurfacenet import FastSDNetSeg, FastSDNetCls


def export_cls_onnx(model, out_path, dummy_input):
    model.eval()
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes = {
        'input': {0: 'batch'},
        'score': {0: 'batch'},
        'feature1': {0: 'batch'},
        'feature2': {0: 'batch'},
        'feature3': {0: 'batch'},
    }
    output_names = ['score', 'feature1', 'feature2', 'feature3']
    # os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, f'{out_path}_{model._get_name()}.onnx',
                      input_names=['input'], output_names=output_names, dynamic_axes=dynamic_axes, verbose=True)
    print(f'out_path: {out_path}_{model._get_name()}.onnx Finished!')
    return


def export_seg_onnx(model, out_path, dummy_input):
    model.eval()
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes = {
        'feature1': {0: 'batch'},
        'feature2': {0: 'batch'},
        'feature3': {0: 'batch'},
        'output': {0: 'batch'}
    }
    input_names = ['feature1', 'feature2', 'feature3']
    output_names = ['output']
    # os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, f'{out_path}_{model._get_name()}.onnx',
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, verbose=True)
    print(f'out_path: {out_path}_{model._get_name()}.onnx Finished!')
    return


def cac_fps(model, size=(256, 256), device='cpu'):
    model.to(device)
    with torch.no_grad():
        model.eval()
        sample = torch.rand(1, 3, *size)
        sample = sample.to(device)
        num = 100
        for i in range(80):
            _ = model(sample)

        start = time.time()
        for i in range(num):
            _ = model(sample)

    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}')
    return fps


def cac_fastsurface_fps(model, size=(256, 256), device='cpu', ratio=0.9):
    model.to(device)
    with torch.no_grad():
        model.eval()
        sample = torch.rand(1, 3, *size)
        sample = sample.to(device)
        num = 100
        for i in range(80):
            _ = model(sample)

        start = time.time()
        for i in range(num):
            e1, e2, e3 = model.feature(sample)
            _ = model.cls(e3)
            if i >= ratio*num:
                out = model.seg(e1, e2, e3)
                out = F.interpolate(out[0], size=size, mode='bilinear', align_corners=True)
    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}')
    return fps


def cac_fastsurface_fps_onnx(infer_cls, infer_seg, size=(256, 256), ratio=0.9):
    sample = np.random.rand(1, 3, *size).astype(np.float32)
    num = 3600
    for i in range(80):
        score, e1, e2, e3 = infer_cls.run(['score', 'feature1', 'feature2', 'feature3'], {'input': sample})
        out = infer_seg.run(['output'], {'feature1': e1, 'feature2': e2, 'feature3': e3})

    start = time.time()
    for i in range(num):
        score, e1, e2, e3 = infer_cls.run(['score', 'feature1', 'feature2', 'feature3'], {'input': sample})
        if i >= ratio*num:
            out = infer_seg.run(['output'], {'feature1': e1, 'feature2': e2, 'feature3': e3})
            out = interpolate_bilinear(out[0], (1, 2, 256, 256))
    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}')
    return fps


def interpolate_bilinear(input, size):
    # 输入大小为 (batch_size, channels, height, width)
    # 输出大小为 (batch_size, channels, out_height, out_width)
    input_shape = input.shape
    batch_size, channels, out_height, out_width = size

    scale_h = (input_shape[2] - 1) / (out_height - 1)
    scale_w = (input_shape[3] - 1) / (out_width - 1)

    h = np.arange(out_height).reshape(-1, 1) * scale_h
    w = np.arange(out_width).reshape(1, -1) * scale_w

    h0 = np.floor(h).astype(np.int32)
    w0 = np.floor(w).astype(np.int32)
    h1 = np.minimum(h0 + 1, input_shape[2] - 1)
    w1 = np.minimum(w0 + 1, input_shape[3] - 1)

    s1 = h - h0
    s0 = 1 - s1
    t1 = w - w0
    t0 = 1 - t1

    s0 = np.expand_dims(s0, axis=-1)
    s1 = np.expand_dims(s1, axis=-1)
    t0 = np.expand_dims(t0, axis=-2)
    t1 = np.expand_dims(t1, axis=-2)

    i0 = input[:, :, h0.flatten(), w0.flatten()] * s0 * t0
    i1 = input[:, :, h1.flatten(), w0.flatten()] * s1 * t0
    i2 = input[:, :, h0.flatten(), w1.flatten()] * s0 * t1
    i3 = input[:, :, h1.flatten(), w1.flatten()] * s1 * t1

    output = i0 + i1 + i2 + i3
    return output.reshape(batch_size, channels, out_height, out_width)


model_name = sys.argv[1]
model = get_model(model_name, class_num=2)
# cls_model = FastSDNetCls(3, 2)
# seg_model = FastSDNetSeg(2)

s = torch.rand(1, 3, 256, 256)
t = (torch.rand(1, 16, 64, 64), torch.rand(1, 32, 32, 32), torch.rand(1, 32, 16, 16))
#export_cls_onnx(cls_model, '', s)
#export_seg_onnx(seg_model, '', t)

infer_cls = onnxruntime.InferenceSession('./_FastSDNetCls.onnx')
infer_seg = onnxruntime.InferenceSession('./_FastSDNetSeg.onnx')
cac_fastsurface_fps_onnx(infer_cls, infer_seg)
# model.eval()
# summary(model, torch.rand((1, 3, 256, 256)))

# if model_name == 'fastsurfacenet':
#     model.mode = 'seg'
#     cac_fastsurface_fps(model)
# else:
#     cac_fps(model)
