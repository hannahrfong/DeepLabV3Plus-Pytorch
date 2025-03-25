import random
import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from ._utils import adaptive_add
from .dynamic_operations import (DynamicBatchNorm2d, DynamicBinConv2d,
                                 DynamicFPLinear, DynamicLearnableBias,
                                 DynamicPReLU, DynamicQConv2d)
from .operations import BinaryActivation
import math
def tuple2cand(cand_tuple):
    return torch.tensor(cand_tuple).reshape(-1, 6)


class BasicBlock(nn.Module):

    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups1_list,
                 groups2_list,
                 wh,
                 stride=1,
                 dilation=1):
        super().__init__()
        assert stride in [1, 2]
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = ks_list

        self.stride = stride
        self.dilation = dilation
        cur_wh = wh

        self.move11 = DynamicLearnableBias(max_inp)
        self.binary_activation1 = BinaryActivation()
        self.binary_conv = DynamicBinConv2d(max_inp,
                                            max_inp,
                                            ks_list,
                                            groups1_list,
                                            wh=cur_wh,
                                            stride=stride,
                                            dilation=dilation)
        self.bn1 = DynamicBatchNorm2d(max_inp)
        self.move12 = DynamicLearnableBias(max_inp)
        self.prelu1 = DynamicPReLU(max_inp)
        self.move13 = DynamicLearnableBias(max_inp)
        self.move21 = DynamicLearnableBias(max_inp)
        self.binary_activation2 = BinaryActivation()
        cur_wh //= stride
        self.binary_conv1x1 = DynamicBinConv2d(max_inp,
                                               max_oup, [1],
                                               groups2_list,
                                               wh=cur_wh)
        self.bn2 = DynamicBatchNorm2d(max_oup)
        self.move22 = DynamicLearnableBias(max_oup)
        self.prelu2 = DynamicPReLU(max_oup)
        self.move23 = DynamicLearnableBias(max_oup)
        self.ops_memory = {}

    def forward(self, x, loss, sub_path):
        # RSign
        print("******** HIHI")
        out1 = self.move11(x)
        print(out1.shape[-1])
        print(x.shape[-1])
        out1 = self.binary_activation1(out1)
        print(out1.shape[-1])
        print(x.shape[-1])
        # Conv
        out1, loss = self.binary_conv(out1,
                                      loss,
                                      sub_path=[-1, sub_path[1], sub_path[2]])
        print(out1.shape[-1])
        print(x.shape[-1])
        out1 = self.bn1(out1)
        print(out1.shape[-1])
        print(x.shape[-1])
        print("! BASIC BLOCK STEP 1 !")
        # shortcut
        out1 = adaptive_add(out1, x)
        # RPReLU
        print("! BASIC BLOCK STEP 2 !")
        out1 = self.move12(out1)
        print("! BASIC BLOCK STEP 3 !")
        out1 = self.prelu1(out1)
        print("! BASIC BLOCK STEP 4 !")
        out1 = self.move13(out1)
        print("! BASIC BLOCK STEP 5 !")
        out2 = self.move21(out1)
        print("! BASIC BLOCK STEP 6 !")
        out2 = self.binary_activation2(out2)
        print("! BASIC BLOCK STEP 7 !")
        out2, loss = self.binary_conv1x1(
            out2, loss, sub_path=[sub_path[0], 1, sub_path[3]])
        print("! BASIC BLOCK STEP 8 !")
        out2 = self.bn2(out2)
        print("! BASIC BLOCK STEP 9 !")
        out2 = adaptive_add(out2, out1)
        print("! BASIC BLOCK STEP 10 !")
        # RPReLU
        out2 = self.move22(out2)
        print("! BASIC BLOCK STEP 11 !")
        out2 = self.prelu2(out2)
        print("! BASIC BLOCK STEP 12 !")
        out2 = self.move23(out2)
        print("! BASIC BLOCK STEP 13 !")
        return out2, loss

    def get_flops_bitops(self, pre_sub_path, sub_path):
        if tuple(pre_sub_path + sub_path) not in self.ops_memory:
            pre_channels, _, _, _ = pre_sub_path
            channels, ks, groups1, groups2 = sub_path
            bitops1 = (ks * ks * pre_channels // groups1 * pre_channels *
                       self.binary_conv.wh // self.stride *
                       self.binary_conv.wh // self.stride)
            bitops2 = (1 * 1 * pre_channels // groups2 * channels *
                       self.binary_conv1x1.wh * self.binary_conv1x1.wh)
            flops = 0.0
            bitops = bitops1 + bitops2
            self.ops_memory[tuple(pre_sub_path +
                                  sub_path)] = flops / 1e6, bitops / 1e6
        return self.ops_memory[tuple(pre_sub_path + sub_path)]

    def to_static(self, x, loss, sub_path):
        # RSign
        out1 = self.move11.to_static(x)
        out1 = self.binary_activation1(out1)
        # Conv
        out1, loss = self.binary_conv.to_static(
            out1, loss, sub_path=[-1, sub_path[1], sub_path[2]])
        out1 = self.bn1.to_static(out1)
        # shortcut
        out1 = adaptive_add(out1, x)
        # RPReLU
        out1 = self.move12.to_static(out1)
        out1 = self.prelu1.to_static(out1)
        out1 = self.move13.to_static(out1)
        out2 = self.move21.to_static(out1)
        out2 = self.binary_activation2(out2)
        out2, loss = self.binary_conv1x1.to_static(
            out2, loss, sub_path=[sub_path[0], 1, sub_path[3]])
        out2 = self.bn2.to_static(out2)
        out2 = adaptive_add(out2, out1)
        # RPReLU
        out2 = self.move22.to_static(out2)
        out2 = self.prelu2.to_static(out2)
        out2 = self.move23.to_static(out2)
        return out2, loss


class StemBlock(nn.Module):

    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups1_list,
                 groups2_list,
                 wh,
                 stride=1,
                 dilation=1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = ks_list

        self.conv = DynamicQConv2d(max_inp,
                                   max_oup,
                                   ks_list,
                                   groups1_list,
                                   w_bit=8,
                                   a_bit=None,
                                   wh=wh,
                                   stride=stride)
        self.bn = DynamicBatchNorm2d(max_oup)
        self.move1 = DynamicLearnableBias(max_oup)
        self.relu = DynamicPReLU(max_oup)
        self.move2 = DynamicLearnableBias(max_oup)

    def forward(self, x, loss, sub_path):
        print("START OF STEP STEM")
        print(x.shape[-1])
        out = self.conv(x, sub_path=sub_path[:3])
        print(out.shape[-1])
        out = self.bn(out)
        print(out.shape[-1])
        out = self.move1(out)
        print(out.shape[-1])
        out = self.relu(out)
        print(out.shape[-1])
        out = self.move2(out)
        print(out.shape[-1])
        print("BEFORE EXITING STEM")
        print(out.shape[-1])
        return out, loss

    def get_flops_bitops(self, sub_path):
        return self.conv.get_flops_bitops(sub_path[:3])

    def to_static(self, x, loss, sub_path):
        out = self.conv.to_static(x, sub_path=sub_path[:3])
        out = self.bn.to_static(out)
        out = self.move1.to_static(out)
        out = self.relu.to_static(out)
        out = self.move2.to_static(out)
        return out, loss


class SuperBNN(nn.Module):

    def __init__(self, cfg, n_class=1000, img_size=224, sub_path=None, replace_stride_with_dilation=None, reduced_channels=None):
        super().__init__()
        self.cfg = cfg

        self.n_class = n_class
        self.img_size = img_size
        self.sub_path = sub_path

        cur_img_size = self.img_size
        self.features = nn.ModuleList()
        self.search_space = []
        self.max_inp = 3

        self.dilation = 1 # start of my changes
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False, False, False]

        self.reduced_channels = reduced_channels or {}

        if len(replace_stride_with_dilation) != 6:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 6-element tuple, got {}".format(replace_stride_with_dilation))
        
        for i, (channels_list, num_blocks_list, ks_list, groups1_list,
                groups2_list, stride) in enumerate(self.cfg):
            max_channels = self.reduced_channels.get(f'stage{i}', max(channels_list))
            # max_ks = max(ks_list)
            max_num_blocks = max(num_blocks_list)
            stage = nn.ModuleList()
            stage_search_space = []
            for j in range(max_num_blocks):
                block = StemBlock if i == 0 else BasicBlock

                if replace_stride_with_dilation[i]:
                    self.dilation += stride
                    stride = 1

                if j == 0:
                    stage.append(
                        block(self.max_inp,
                              max_channels,
                              ks_list,
                              groups1_list,
                              groups2_list,
                              wh=cur_img_size,
                              stride=stride, 
                              dilation=self.dilation))
                    self.max_inp = max_channels
                    cur_img_size //= stride
                    #cur_img_size = math.ceil(cur_img_size / stride)
                else:
                    stage.append(
                        BasicBlock(self.max_inp,
                                   max_channels,
                                   ks_list,
                                   groups1_list,
                                   groups2_list,
                                   wh=cur_img_size))
                stage_search_space.append(
                    [channels_list, ks_list, groups1_list, groups2_list])
            self.features.append(stage)
            self.search_space.append([stage_search_space, num_blocks_list])
       #self.globalpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = DynamicFPLinear(self.max_inp, n_class)
        self.set_bin_weight()
        self.set_bin_activation()
        self.close_distill()
        self.register_buffer('biggest_cand', self.get_biggest_cand())
        self.register_buffer('smallest_cand', self.get_smallest_cand())
        self._output_channels = cfg[-1][0][0]  # Last stage's first channel
        self._low_level_channels = cfg[1][0][0]  # First relevant stage        
        _, _, self.biggest_ops = self.get_ops(self.biggest_cand)
        _, _, self.smallest_ops = self.get_ops(self.smallest_cand)

        # Debug: Print layer names
        print("\nNAS-BNN Layer Structure:")
        for stage_idx, stage in enumerate(self.features):
            for block_idx, block in enumerate(stage):
                print(f"Stage {stage_idx}, Block {block_idx}: features.{stage_idx}.{block_idx}")

    def forward(self, x, sub_path=None):
        out = OrderedDict()
        loss = 0.
        if sub_path is None:
            assert self.sub_path is not None
            sub_path = self.sub_path
        for i, j, channels, ks, groups1, groups2 in sub_path:
            print("WORLD")
            print(i)
            print(j)
            #print(self.features[i][j])
            if i == -1 or j == -1:
                continue
            x, loss = self.features[i][j](
                x, loss,
                [channels.item(),
                 ks.item(),
                 groups1.item(),
                 groups2.item()])

            if i == 1:
                out['low_level'] = x
            elif i == 5:
                out['out'] = x
        return out

    # def forward(self, x, sub_path=None):
    #     out = OrderedDict()
    #     loss = 0.0
        
    #     # Use default sub_path if not provided
    #     if sub_path is None:
    #         sub_path = self.sub_path
    #         print(f"Using default sub_path: {sub_path}")

    #     # Debug: Print input shape
    #     print(f"Input shape: {x.shape}")
        
    #     # Process each layer in the sub_path
    #     for layer_idx, (stage_idx, block_idx, channels, ks, groups1, groups2) in enumerate(sub_path):
    #         # Skip invalid entries
    #         if stage_idx == -1 or block_idx == -1:
    #             print(f"Skipping invalid layer {layer_idx}: stage={stage_idx}, block={block_idx}")
    #             continue

    #         # Debug: Print current processing step
    #         print(f"Processing stage {stage_idx} block {block_idx} | Layer {layer_idx+1}/{len(sub_path)}")
            
    #         # Get the corresponding block
    #         block = self.features[stage_idx][block_idx]
            
    #         # Forward pass through the block
    #         x, loss = block(
    #             x, loss,
    #             [
    #                 channels.item(),
    #                 ks.item(),
    #                 groups1.item(),
    #                 groups2.item()
    #             ]
    #         )
            
    #         # Capture low-level features after stage 1, block 0
    #         if stage_idx == 1 and block_idx == 0:
    #             out['low_level'] = x
    #             print(f"Captured low_level feature at stage {stage_idx} block {block_idx} | Shape: {x.shape}")
            
    #         # Capture final output features at last stage
    #         if stage_idx == len(self.cfg) - 1:
    #             out['out'] = x
    #             print(f"Captured final output at stage {stage_idx} | Shape: {x.shape}")

    #     # Debug: Verify captured features
    #     print("Captured features:", list(out.keys()))
    #     print("Feature shapes:", {k: v.shape for k, v in out.items()})
        
    #     # Verify required features exist
    #     if 'low_level' not in out:
    #         raise ValueError("low_level feature not captured in forward pass")
    #     if 'out' not in out:
    #         raise ValueError("out feature not captured in forward pass")
    #     print("Captured features:", list(out.keys()))
    #     return out
    
    def get_output_channels(self):
        return self.reduced_channels.get(f'stage{len(self.cfg)-1}', max(self.cfg[-1][0][0]))

    def get_lowlevel_channels(self):
        return self.reduced_channels.get('stage1', max(self.cfg[1][0][0]))
    
    def get_ops(self, sub_path=None):
        if sub_path is None:
            assert self.sub_path is not None
            sub_path = self.sub_path
        flops = 0.0
        bitops = 0.0
        pre = None
        cur = None
        for i, j, channels, ks, groups1, groups2 in sub_path:
            if i == -1 or j == -1:
                continue
            cur = [channels.item(), ks.item(), groups1.item(), groups2.item()]
            if i > 0:
                tmp_flops, tmp_bitops = self.features[i][j].get_flops_bitops(
                    pre, cur)
            else:
                tmp_flops, tmp_bitops = self.features[i][j].get_flops_bitops(
                    cur)
            pre = cur
            flops += tmp_flops
            bitops += tmp_bitops
        #tmp_flops, tmp_bitops = self.fc.get_flops_bitops(pre)
        flops += tmp_flops
        bitops += tmp_bitops
        ops = flops + bitops / 64
        return flops, bitops, ops

    def set_fp_weight(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                m.is_bin = False

    def set_fp_weight_prob(self, prob):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                if np.random.random_sample() <= prob:
                    m.is_bin = False

    def set_bin_weight(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                m.is_bin = True

    def set_bin_activation(self):
        for m in self.modules():
            if isinstance(m, BinaryActivation):
                m.is_bin = True

    def set_fp_activation(self):
        for m in self.modules():
            if isinstance(m, BinaryActivation):
                m.is_bin = False

    def set_fp_activation_prob(self, prob):
        for m in self.modules():
            if isinstance(m, BinaryActivation):
                if np.random.random_sample() <= prob:
                    m.is_bin = False

    def open_distill(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                m.is_distill = True

    def close_distill(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                m.is_distill = False

    def to_static(self, x):
        loss = 0.
        assert self.sub_path is not None
        for i, j, channels, ks, groups1, groups2 in self.sub_path:
            if i == -1 or j == -1:
                continue
            x, loss = self.features[i][j].to_static(
                x, loss,
                [channels.item(),
                 ks.item(),
                 groups1.item(),
                 groups2.item()])
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc.to_static(x)
        return x, loss

    def get_random_cand(self):
        if hasattr(self, 'module'):
            m = self.module
        else:
            m = self
        device = next(m.parameters()).device
        res = []
        for i, stage_cand in enumerate(self.search_space):
            block_num = random.choice(stage_cand[1])
            for j, block_cand in enumerate(stage_cand[0]):
                if j >= block_num:
                    cur = [-1, -1]
                else:
                    cur = [i, j]
                if len(res) == 0:
                    last_channel = -1
                else:
                    last_channel = res[-1][0, 2]
                for idx, k in enumerate(block_cand):
                    if idx == 0:
                        new_k = torch.tensor(k)
                        new_k = new_k[new_k >= last_channel].tolist()
                    else:
                        new_k = k
                    cur += [random.choice(new_k)]
                res.append(torch.tensor(cur)[None, :])
        res = torch.cat(res, dim=0).to(device)
        return res

    def get_random_range_cand(self, ops_min, ops_max):
        found = False
        while not found:
            res = self.get_random_cand()
            _, _, ops = self.get_ops(res)
            if ops_min <= ops <= ops_max:
                found = True
        return res

    def get_biggest_cand(self):
        res = []
        for i, stage_cand in enumerate(self.search_space):
            block_num = max(stage_cand[1])
            for j, block_cand in enumerate(stage_cand[0]):
                if j >= block_num:
                    cur = [-1, -1]
                else:
                    cur = [i, j]
                for k, layer_cand in enumerate(block_cand):
                    if k == len(block_cand) - 1 or k == len(block_cand) - 2:
                        cur += [min(layer_cand)]
                    else:
                        cur += [max(layer_cand)]
                res.append(torch.tensor(cur)[None, :])
        res = torch.cat(res, dim=0)
        return res

    def get_smallest_cand(self):
        res = []
        for i, stage_cand in enumerate(self.search_space):
            block_num = min(stage_cand[1])
            for j, block_cand in enumerate(stage_cand[0]):
                if j >= block_num:
                    cur = [-1, -1]
                else:
                    cur = [i, j]
                for k, layer_cand in enumerate(block_cand):
                    if k == len(block_cand) - 1 or k == len(block_cand) - 2:
                        cur += [max(layer_cand)]
                    else:
                        cur += [min(layer_cand)]
                res.append(torch.tensor(cur)[None, :])
        res = torch.cat(res, dim=0)
        return res

def nasbnn(**kwargs):
    cfg = [[[24, 32, 48], [1], [3], [1], [1], 2],
           [[48, 64, 96], [2, 3], [3], [1], [1], 1],
           [[96, 128, 192], [2, 3], [3, 5], [1, 2], [1], 2],
           [[192, 256, 384], [2, 3], [3, 5], [2, 4], [1], 2],
           [[384, 512, 768], [8, 9], [3, 5], [4, 8], [1], 2],
           [[768, 1024, 1536], [2, 3], [3, 5], [8, 16], [1], 2]]

    checkpoint_path = os.path.join(os.path.dirname(__file__), "info.pth.tar")
    pretrained_path = os.path.join(os.path.dirname(__file__), "checkpoint.pth.tar")  # Path to pretrained weights

    # Create model architecture
    searched = torch.load(checkpoint_path)
    ops = 140
    sub_path = searched['pareto_global'][ops]
    model = SuperBNN(cfg, 1000, 513, tuple2cand(sub_path),**kwargs)

    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        new_key = k.replace("module.", "")  # Remove 'module.' if needed
        state_dict[new_key] = v

    model.load_state_dict(state_dict, strict=False)   
    
    return model  # Return the initialized model
