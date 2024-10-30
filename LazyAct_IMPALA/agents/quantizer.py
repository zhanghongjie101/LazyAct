import os, sys

import torch
import torch.nn as nn
from torch import quantization
import time
from copy import deepcopy

class Quantizer(object):
    def __init__(self, args, train_steps, model_filepath, quant_model_filepath):
        self.args = args
        self.train_steps = train_steps
        self.model_filepath = model_filepath
        self.quant_model_filepath = quant_model_filepath
        self.layers = [['conv1','relu1'], ['conv2','relu2'], ['conv3','relu3'], ['fc', 'relu4']]
        self.types_to_quantize = {nn.Conv2d, nn.ReLU, nn.Linear}

    def quant_and_save(self):
        quant = None
        while self.train_steps.value<=self.args.total_num_steps:
            time.sleep(0.1)
            while True:
                try:
                    quant = torch.load(self.model_filepath, map_location=torch.device('cpu'))
                    break
                except:
                    time.sleep(0.01)
            quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            f = quantization.fuse_modules(quant, self.layers, inplace=True)
            actor_quant = quantization.quantize_dynamic(f, self.types_to_quantize, dtype=torch.qint8)
            torch.save(actor_quant, self.quant_model_filepath)
