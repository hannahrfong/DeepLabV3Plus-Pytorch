from tqdm import tqdm
import network
import utils
import os
import random
import argparse
from collections import OrderedDict
import numpy as np
from copy import deepcopy

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from network.backbone.dynamic_operations import (DynamicBatchNorm2d, DynamicBinConv2d,
                                 DynamicFPLinear, DynamicLearnableBias,
                                 DynamicPReLU, DynamicQConv2d)
from network.backbone.nasbnn import SuperBNN, StemBlock
from network._deeplab import DeepLabHeadV3Plus, ASPP, ASPPConv
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
class BNNPruner:
    def __init__(self, model, threshold=2):
        self.model = model
        self.threshold = threshold
        
        # Initialize tracking structures (Replaces flip_mat_sum)
        self.flip_counts = []
        self.target_modules = []
        name_list = []
        for name, module in model.named_modules():
            parts = name.split('.')
            concat_name = '.'.join(parts[:5])
            if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)) and (concat_name not in name_list):  
                name_list.append(concat_name)
                self.target_modules.append(module.weight)
                self.flip_counts.append(torch.zeros_like(module.weight.data))
        
        # Replaces target_modules_last
        self.prev_weights = [m.data.clone() for m in self.target_modules]

    def track_flips(self, model):
        n = 0
        name_list = []
        for name, module in model.named_modules():
            parts = name.split('.')
            concat_name = '.'.join(parts[:5])
            if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)) and (concat_name not in name_list):
                self.target_modules[n] = module.weight
                name_list.append(concat_name)
                n = n + 1

        print("\n===== Tracking Weight Flips =====")
        for i, (weight, prev_weight) in enumerate(zip(self.target_modules, self.prev_weights)):
            # Calculate flips (0->1 or 1->0)
            current_bin = torch.sign(weight.data)
            prev_bin = torch.sign(prev_weight)
            flip_mask = (current_bin != prev_bin).float()  # Actual sign flips
            self.flip_counts[i] += flip_mask
            
            # Debug: Print layer flip stats
            total_flips = flip_mask.sum().item()
            print(f"Layer {i}: Total flips = {total_flips}")
            
            # Update previous weights
            self.prev_weights[i] = weight.data.clone()
    
    def compute_p_L(self):
        print("\n===== Computing p_L (Weight-Level) =====")
        p_L = {}
        sorted_indices = []
        for layer_idx, flip_counts in enumerate(self.flip_counts):
            if flip_counts.dim() < 4:  # Skip non-conv layers (e.g., FC)
                print("RETURNED HERE")
                continue
             # Debug: Print raw flip counts for the first few weights
            p_L_file.write(f"\nLayer {layer_idx} - Raw Flip Counts:")
            p_L_file.write(f" Shape: {flip_counts.shape}")
            
            # Print a subset of the tensor (e.g., first 3 filters, first 3 channels)
            # subset = flip_counts[:3, :3, :2, :2]  # Adjust indices as needed
            # print("Sample values (first 3 filters, 3 channels, 2x2 kernel):\n", subset)
            
            # Optional: Print statistics
            p_L_file.write(f" Min flips: {flip_counts.min().item()}, Max flips: {flip_counts.max().item()}")
            p_L_file.write(f" Mean flips: {flip_counts.float().mean().item():.2f}")

            # 1. Calculate total insensitive WEIGHTS in this layer
            insensitive_weights_mask = (flip_counts >= self.threshold)
            total_insensitive = insensitive_weights_mask.sum().item()
            total_weights = flip_counts.numel()
            p_L[layer_idx] = (total_insensitive / total_weights) * 100
            
            # 2. Calculate insensitive weights PER CHANNEL
            insensitive_per_channel = insensitive_weights_mask.sum(dim=(1, 2, 3))  # [out_channels]
            
            # 3. Sort channels by number of insensitive weights (descending)
            sorted_idx = torch.argsort(insensitive_per_channel, descending=True)
            sorted_indices.append(sorted_idx)
            
            # Debug: Print layer statistics
            out_channels = flip_counts.shape[0]
            print(f"Layer {layer_idx} (Output Channels: {out_channels}):")
            print(f"  Insensitive Weights = {total_insensitive}/{total_weights} ({p_L[layer_idx]:.1f}%)")
            print(f"  Channels Sorted by Insensitive Weights: {sorted_idx.tolist()[:5]}...")  # Top 5
            print(sorted_indices[layer_idx].size())
        p_L_file.flush()
        return p_L, sorted_indices
    
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_nasbnn',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    
    # Pruning Options
    parser.add_argument('--prune', action='store_true', 
                       help='Enable BNN pruning using weight flipping frequency')
    parser.add_argument('--prune_train_iterations', type=int, default=5,
                       help='Number of epochs for training after pruning phase')
    parser.add_argument('--prune_retrain_iterations', type=int, default=5,
                       help='Number of epochs for retraining pre-trained model before pruning phase')
    parser.add_argument('--prune_threshold', type=int, default=2,
                       help='Flip frequency threshold for pruning')

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    model.eval() 
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

# [
# [[48], [1], [3], [1], [1], 2], 
# [[96], [2, 3], [3], [1], [1], 1],
#  [[192], [2, 3], [3, 5], [1, 2], [1], 2],
#  [[384], [2, 3], [3, 5], [2, 4], [1], 2],
#  [[768], [8, 9], [3, 5], [4, 8], [1], 2],
#  [[768, 1024, 1536], [2, 3], [3, 5], [8, 16], [1], 2]]


def rebuild_pruned_nasbnn(model, p_L, sorted_indices, device, opts, make_new_model = False):
    print("\n===== Rebuilding Pruned NAS-BNN =====")
    # Unwrap DataParallel and get backbone
    if isinstance(model, nn.DataParallel):
        orig_model = model.module
    original_backbone = orig_model.backbone
    original_classifier = orig_model.classifier
    name_list = []
    # 1. Identify all DynamicBinConv2d layers and their stage indices
    conv_layers = []
    for name, module in orig_model.named_modules(): 
        parts = name.split('.')    
        concat_name = '.'.join(parts[:4])
        if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)) and concat_name not in name_list:  
            name_list.append(concat_name)
            stage_idx = int(parts[2])            
            block_idx = int(parts[3])
            conv_layers.append((stage_idx, block_idx, name))
    
    start_range = 0
    end_range = len(conv_layers) 
    tracked_layers = conv_layers[start_range:end_range+1]  # [(stage_idx, layer_name), ...]
    
    # Debug: Print tracked layers
    print("Tracked Layers for Pruning:")
    for idx, (stage_idx, block_idx, name) in enumerate(tracked_layers):
        print(f"  Layer {idx}: Stage {stage_idx}, Block Index: {block_idx} - {name}")
    
    # 3. Clone and modify cfg
    cfg = deepcopy(original_backbone.cfg)

    # Track max new_max per stage
    stage_max_new_max = {}
    # First pass: compute new_max for each layer and track per stage
    for layer_idx, (stage_idx, block_idx, layer_name) in enumerate(tracked_layers):
        p_L_file.write(f"Original Layer Index: {layer_idx}, Stage Index: {stage_idx}, Block Index: {block_idx}, Layer Name: {layer_name}\n")

        if layer_idx >= len(p_L):  # Ensure p_L has entries for all tracked layers
            continue
        
        # Get original channel list 
        layer_module = orig_model.backbone.features[stage_idx][block_idx]
        if isinstance(layer_module, StemBlock):
            conv = layer_module.conv
            original_max = layer_module.conv.weight.shape[0]  # Output channels
        else:
            conv = layer_module.binary_conv
            original_max = layer_module.binary_conv.weight.shape[0]  # Output channels
        prune_ratio = p_L[layer_idx] / 100.0
        num_pruned = int(original_max * prune_ratio)
        num_pruned = min(num_pruned, original_max - 1)
        
        # Use sorted_indices to determine kept channels
        kept_channels = sorted_indices[layer_idx][num_pruned:].tolist()
        conv.weight = nn.Parameter(conv.weight[kept_channels,:,:,:])



        #-------------only for making a new model----------------------   

        new_max = original_max - num_pruned
        new_max = max(new_max, 1)

        if stage_idx not in stage_max_new_max or new_max > stage_max_new_max[stage_idx]:
            stage_max_new_max[stage_idx] = new_max
        else:
            stage_max_new_max[stage_idx] = max(stage_max_new_max[stage_idx], new_max)
        
        # Update groups1 and groups2 to valid divisors of new_max
        original_groups1 = cfg[stage_idx][3]
        original_groups2 = cfg[stage_idx][4]
        
        # Filter groups1 to valid divisors of new_max
        valid_groups1 = [g for g in original_groups1 if new_max % g == 0]
        if not valid_groups1:
            valid_groups1 = [1]  # Ensure at least one group
        cfg[stage_idx][3] = valid_groups1
        
        # Filter groups2 similarly
        valid_groups2 = [g for g in original_groups2 if new_max % g == 0]
        if not valid_groups2:
            valid_groups2 = [1]
        cfg[stage_idx][4] = valid_groups2
        
        p_L_file.write(f"Stage {stage_idx}, Layer: {layer_idx}, Block: {block_idx}: Pruned {num_pruned}/{original_max} channels | Conv {conv.weight.shape[0]}\n")
        print(f"Stage {stage_idx}, Layer: {layer_idx}, Block: {block_idx}: Pruned {num_pruned}/{original_max} channels | Conv {conv.weight.shape[0]}\n")
        # print(f"  Updated groups1: {valid_groups1}, groups2: {valid_groups2}\n")
    

    if not make_new_model:
        return model

    # Second pass: Update cfg with max_new_max and valid groups per stage
    # for stage_idx in stage_max_new_max:
    #     new_max = stage_max_new_max[stage_idx]
    #     cfg[stage_idx][0] = [new_max]
    #     print(f"\nFinal Stage {stage_idx}: new_max = {new_max}")
    #     print(f"  Updated groups1: {cfg[stage_idx][3]}, groups2: {cfg[stage_idx][4]}")
    #     print(f"  Updated groups1: {valid_groups1}, groups2: {valid_groups2}\n")
        
    if opts.output_stride==8:
        replace_stride_with_dilation = [False, False, False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, False, False, False, True]
        aspp_dilate = [6, 12, 18]




    pruned_model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    name_list = []
    # Replace just the backbone with our pruned version
    pruned_model.backbone = SuperBNN(
        cfg,
        img_size=opts.crop_size,
        replace_stride_with_dilation=replace_stride_with_dilation
    )

    sub_path = []
    for stage_idx, stage_cfg in enumerate(pruned_model.backbone.cfg):
        channels_list, num_blocks_list, ks_list, groups1_list, groups2_list, stride = stage_cfg
        max_channels = max(channels_list)
        max_ks = max(ks_list)
        max_groups1 = max(groups1_list)
        max_groups2 = max(groups2_list)
        
        for block_idx in range(max(num_blocks_list)):
            sub_path.append([
                stage_idx, block_idx, 
                max_channels, max_ks, 
                max_groups1, max_groups2
            ])
    
    new_inplanes = max(cfg[-1][0])  # Last stage's max channel
    new_low_level_planes = max(cfg[1][0])  # Stage 1's max channel

    pruned_model.backbone.sub_path = sub_path=torch.tensor(sub_path).to(device)
    pruned_model.classifier = DeepLabHeadV3Plus(new_inplanes, new_low_level_planes, opts.num_classes, aspp_dilate)
    utils.set_bn_momentum(pruned_model.backbone, momentum=0.01)

    print("Generated sub_path for pruned model:", pruned_model.backbone.sub_path.shape)

    for name, module in pruned_model.backbone.named_modules():
        if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)):
            parts = name.split('.')
            stage_idx = int(parts[1])            
            block_idx = int(parts[2])

    start_range = 0
    end_range = len(conv_layers) 
    tracked_layers = conv_layers[start_range:end_range+1]  # [(stage_idx, layer_name), ...]
    
    # Debug: Print tracked layers
    p_L_file.write("\nAfter Layers for Pruning:\n")
    for layer_idx, (stage_idx, block_idx, layer_name) in enumerate(tracked_layers):
        layer_module = pruned_model.backbone.features[stage_idx][block_idx]
        if isinstance(layer_module, StemBlock):
            pruned_max = layer_module.conv.weight.data.shape[0]  # Output channels
        else:
            pruned_max = layer_module.binary_conv.weight.data.shape[0]  # Output channels
        p_L_file.write(f"Layer Index: {layer_idx}, Stage Index: {stage_idx}, Block Index: {block_idx}, Layer Name: {layer_name} | Pruned Max: {pruned_max}\n")
    p_L_file.flush()
    print("Pruned NAS-BNN rebuilt. Retrain from scratch.")
    return pruned_model

def train_epoch(pruned_model, loader, optimizer, criterion, metrics, scheduler, device, opts, cur_itrs, pruner=None):
    pruned_model.train()
    best_score = 0.0
    if pruner:
        total_itrs = opts.prune_retrain_iterations
    else:
        total_itrs = opts.prune_train_iterations
    print("!!! TOTAL ITERATIONs !!!!: " + str(total_itrs))
    for images, labels in loader:
        cur_itrs += 1
        print("!!! CURRENT ITERATION? !!!!: " + str(cur_itrs))
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = pruned_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if pruner:
            print(f"PRUNER TRACKING FLIPS Iteration {cur_itrs}")
            pruner.track_flips(pruned_model)

        if not pruner and cur_itrs % 100 == 0:
            val_score, ret_samples = validate(opts, pruned_model, loader, device, metrics)
            print(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                path = 'checkpoints/pruned_best_deeplabv3plus_nasbnn_voc_os16.pth'
                torch.save({
                    "cur_itrs": cur_itrs,
                    "model_state": pruned_model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
                    "cfg": pruned_model.module.backbone.cfg,  # Save the pruned cfg
                    "is_pruned": True,  # Flag to indicate pruned model
                }, path)

                print(f"Model saved as {path}")

        if cur_itrs >= total_itrs:
            return

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        print("VAL BATCH SIZE")
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        if checkpoint.get("is_pruned", False):
            print("IS PRUNED")
            # Rebuild pruned model using saved cfg
            pruned_cfg = checkpoint["cfg"]
            #model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
            if opts.output_stride==8:
                replace_stride_with_dilation = [False, False, False, False, True, True]
                aspp_dilate = [12, 24, 36]
            else:
                replace_stride_with_dilation = [False, False, False, False, False, True]
                aspp_dilate = [6, 12, 18]
            model.backbone = SuperBNN(
                pruned_cfg,
                img_size=opts.crop_size,
                replace_stride_with_dilation=replace_stride_with_dilation
            )

            new_inplanes = max(pruned_cfg[-1][0])  # Last stage's max channel
            new_low_level_planes = max(pruned_cfg[1][0])  # Stage 1's max channel
            model.classifier = DeepLabHeadV3Plus(new_inplanes, new_low_level_planes, opts.num_classes, aspp_dilate)
            utils.set_bn_momentum(model.backbone, momentum=0.01)

        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            #optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        print(checkpoint["cur_itrs"])
        print(checkpoint['best_score'])
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #===========PRUNING==============#
    cur_pL = 100.0
    average_pL = 0.0
    if opts.prune:
        while True:
            print("PRUNE ONLY")
            # Initialize pruner
            pruner = BNNPruner(model, opts.prune_threshold)
            
            # Phase 1: Track flips
            print("==> Tracking weight flips...")
            train_epoch(model, train_loader, optimizer, criterion, metrics, scheduler, device, opts, cur_itrs, pruner=pruner)
            #pruner.track_flips(model)
            
            # Phase 2: Compute pruning percentages
            print("==> Computing Pruning Percentages...")
            p_L, sorted_channels = pruner.compute_p_L()
            p_L_file.write(f"\np_L: {p_L}\n")

            average_pL = sum(p_L.values()) / len(p_L) if data else 0
            p_L_file.write(f"\nAverage p_L: {average_pL}\n")
            p_L_file.flush()
            print(f"\nCurrent p_L:{cur_pL} Average p_L: {average_pL}\n")
            if (cur_pL >= average_pL) and average_pL > 0.5:
                cur_pL = average_pL
                # Phase 3: Rebuild pruned model
                print("==> Pruning Original Model...")
                model = rebuild_pruned_nasbnn(model, p_L, sorted_channels, device, opts, make_new_model = False)
            else:
                print("==> Rebuilding Pruning Model...")
                pruned_model = rebuild_pruned_nasbnn(model, p_L, sorted_channels, device, opts, make_new_model = True)
                pruned_model = nn.DataParallel(pruned_model)
                pruned_model.to(device)
                break

        # Phase 4: Retrain
        print("==> Retraining pruned model...")
        cur_itrs = 0
        optimizer = torch.optim.SGD(params=[
                        {'params': pruned_model.module.backbone.parameters(), 'lr': 0.1 * opts.lr},
                        {'params': pruned_model.module.classifier.parameters(), 'lr': opts.lr},
                        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

        train_epoch(pruned_model, train_loader, optimizer, criterion, metrics, scheduler, device, opts, cur_itrs)
        return

    #===========PRUNING==============#



    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            print("!!! CURRENT ITERATION? !!!!: " + str(cur_itrs))

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            print("! TRAIN LOOP STEP 1 !")
            optimizer.zero_grad()
            print("! TRAIN LOOP STEP 2 !")
            outputs = model(images)
            print("! TRAIN LOOP STEP 3 !")
            loss = criterion(outputs, labels)
            print("! TRAIN LOOP STEP 4 !")
            loss.backward()
            print("! TRAIN LOOP STEP 5 !")
            optimizer.step()
            print("! TRAIN LOOP STEP 6 !")
            np_loss = loss.detach().cpu().numpy()
            print("NP_LOSS: " + str(np_loss))
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            print("! TRAIN LOOP STEP 7 !")
            scheduler.step()
            print("! TRAIN LOOP STEP 8 !")
            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    p_L_file = open('pruning_percentages.txt', 'w')
    main()
