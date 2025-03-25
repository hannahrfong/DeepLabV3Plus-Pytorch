# pruning_utils.py
import torch
import numpy as np

class BNNPruner:
    def __init__(self, model, device, metrics, opts, prune_epochs=5, threshold=2):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.opts = opts
        self.model = model
        self.prune_epochs = prune_epochs
        self.threshold = threshold
        self.flip_counts = {}
        
    def track_weight_flips(self, model):
        for name, module in model.named_modules():
            if isinstance(module, DynamicBinConv2d):
                module.track_flips()
                
    def analyze_flips(self):
        flip_counts = {}
        for name, module in self.model.named_modules():
            if isinstance(module, DynamicBinConv2d):
                flip_counts[name] = module.flip_count.cpu().numpy()
        return flip_counts
    
    def compute_pruning_percentages(self, flip_counts):
        p_L = {}
        for name, counts in flip_counts.items():
            total = counts.size
            insensitive = (counts >= self.threshold).sum()
            p_L[name] = (insensitive / total) * 100
        return p_L
    
    def apply_pruning(self, p_L):
        pruned_model = deepcopy(self.model)
        pruned_model.prune_channels(p_L)
        return pruned_model
    
    def retrain(self, model, train_loader, val_loader, opts):
        # Setup optimizer for pruned model
        optimizer = torch.optim.SGD([
            {'params': model.backbone.parameters(), 'lr': 0.01 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr}
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        
        # Short retraining phase
        for epoch in range(self.prune_epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_score = validate(opts, model, val_loader, device, metrics)
            print(f"Pruned model performance: {val_score}")