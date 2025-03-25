from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from copy import deepcopy

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from network.backbone.dynamic_operations import (DynamicBatchNorm2d, DynamicBinConv2d,
                                 DynamicFPLinear, DynamicLearnableBias,
                                 DynamicPReLU, DynamicQConv2d)
from network.backbone.nasbnn import SuperBNN
from network._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

class BNNPruner:
    def __init__(self, model, threshold=2):
        self.model = model
        self.threshold = threshold
        
        # Initialize tracking structures (Replaces flip_mat_sum)
        self.flip_counts = []
        self.target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)):
                self.target_modules.append(module.weight)
                self.flip_counts.append(torch.zeros_like(module.weight.data))
        
        # Replaces target_modules_last
        self.prev_weights = [m.data.clone() for m in self.target_modules]

    def track_flips(self):
        # Update flip counts (Replaces flip_mat accumulation)
        for i, weight in enumerate(self.target_modules):
            current = weight.data
            self.flip_counts[i] += (self.prev_weights[i] != current).float()
            self.prev_weights[i] = current.clone() 
    
    def compute_p_L(self):
        # Calculate pruning percentages per layer
        p_L = {}
        sorted_indices = []
        for i, flips in enumerate(self.flip_counts):
            if flips.dim() < 4: continue  # Skip non-conv layers
            
            # Sum flips per output channel [Critical]
            channel_flips = flips.sum(dim=(1,2,3))  # shape: [out_channels]
            
            # Sort channels by flips (high=insensitive)
            sorted_idx = torch.argsort(channel_flips, descending=True)
            sorted_indices.append(sorted_idx)
            
            # Compute p_L for this layer
            total = channel_flips.numel()
            insensitive = (channel_flips >= self.threshold).sum().item()
            p_L[i] = (insensitive / total) * 100
            
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
    parser.add_argument("--total_itrs", type=int, default=1e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
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
    parser.add_argument('--prune_epochs', type=int, default=5,
                       help='Number of epochs for pruning phase')
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

def get_stage_index(cfg, name):
    # Maps layer name to NAS-BNN cfg index
    parts = name.split('.')
    if 'features' in parts:
        stage = int(parts[parts.index('features')+1])
        return min(stage, len(cfg)-1)
    return 0  # Default to first stage

def rebuild_pruned_nasbnn(orig_model, p_L, sorted_indices):
    cfg = deepcopy(orig_model.cfg)
    
    for i, (name, module) in enumerate(orig_model.named_modules()):
        if isinstance(module, (DynamicBinConv2d, DynamicQConv2d)) and i in p_L:
            # Get original parameters
            stage_idx = get_stage_index(cfg, name)
            orig_channels = cfg[stage_idx][0][0]  # channels_list
            
            # Calculate channels to keep using sorted indices
            num_prune = int(orig_channels * p_L[i]/100)
            keep_indices = sorted_indices[i][num_prune:]  # KEY FIX
            
            # Update config with actual kept channels
            cfg[stage_idx][0][0] = len(keep_indices)  # New channel count
            
    # Rebuild model with exact channel counts
    pruned_model = SuperBNN(cfg, reduced_channels=cfg)
    transfer_weights(orig_model, pruned_model, keep_indices)
    return pruned_model

# Modify transfer_weights function:
def transfer_weights(old_model, new_model, keep_indices):
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    
    for name, param in new_sd.items():
        if 'conv' in name and 'weight' in name:
            stage = get_stage_index(new_model.cfg, name)
            new_sd[name] = old_sd[name][keep_indices[stage]]  # Prune output
            if 'bias' in new_sd:  # Handle biases
                new_sd[name.replace('weight','bias')] = \
                    old_sd[name.replace('weight','bias')][keep_indices[stage]]
        else:
            new_sd[name] = old_sd[name]
    new_model.load_state_dict(new_sd)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

    #===========PRUNING==============#

    if opts.prune:
        print("PRUNE ONLY")
        if not opts.ckpt:
            raise ValueError("--ckpt must be specified for pruning")
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        # 4. Wrap with DataParallel AFTER loading
        model = nn.DataParallel(model)
        model = model.to(device)

        # Initialize pruner
        pruner = BNNPruner(model, opts.prune_threshold)  # Ensure device is not passed here if not in __init__
        
        # Phase 1: Track flips
        print("==> Tracking weight flips...")
        for epoch in range(opts.prune_epochs):
            print(f"TRACKING WEIGHT FLIPS Epoch {epoch}")
            train_epoch(model, train_loader, optimizer, criterion, device)
            pruner.track_flips()
        
        # Phase 2: Compute pruning percentages
        print("==> Computing Pruning Percentages...")
        p_L, sorted_channels = pruner.compute_p_L()
        
        # Phase 3: Rebuild pruned model
        print("==> Rebuilding Pruning Model...")
        pruned_model = rebuild_pruned_nasbnn(model, p_L, sorted_channels, device)
        pruned_model = pruned_model.to(device)
        
        # Phase 4: Retrain
        print("==> Retraining pruned model...")
        optimizer = torch.optim.SGD(pruned_model.parameters(), lr=opts.lr, momentum=0.9)
        for epoch in range(opts.total_itrs):
            print(f"RETRAINING PRUNED MODEL Epoch {epoch}")
            train_epoch(pruned_model, train_loader, optimizer, criterion, device)
            validate(opts, pruned_model, val_loader, device, metrics)
        
        torch.save(pruned_model.state_dict(), "pruned_deeplabv3plus.pth")
        return

    #===========PRUNING==============#

    
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)



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
    main()
