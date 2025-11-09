import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=1.5, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class Trainer:
    """Training and evaluation class - optimized for >70% precision/recall"""
    
    def __init__(self, model, device, model_name='qml_model', class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_precisions = []
        self.train_recalls = []
        self.val_precisions = []
        self.val_recalls = []
        self.train_f1s = []
        self.val_f1s = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_balanced_f1 = 0.0
        self.best_model_state = None
        self.best_threshold = 0.5
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
    
    def find_optimal_threshold(self, y_true, y_probs, min_precision=0.90, min_recall=0.90):
        """Find threshold optimized for >90% precision and recall"""
        thresholds = np.arange(0.1, 0.95, 0.01)  # More granular search
        best_score = -1
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        # First: Try to find threshold where BOTH precision and recall >= 90%
        for threshold in thresholds:
            y_pred_thresh = (y_probs >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) < 2:
                continue
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
            
            if prec >= min_precision and rec >= min_recall:
                # Both criteria met - highest priority
                score = f1 * 5.0  # Very high weight
                # Extra bonus for exceeding targets
                if prec >= 0.95 and rec >= 0.95:
                    score *= 2.0
                elif prec >= 0.92 and rec >= 0.92:
                    score *= 1.5
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_precision = prec
                    best_recall = rec
        
        # If no threshold meets both, prioritize balanced high performance
        if best_score < 0:
            for threshold in thresholds:
                y_pred_thresh = (y_probs >= threshold).astype(int)
                if len(np.unique(y_pred_thresh)) < 2:
                    continue
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
                
                # Balanced scoring for >90% target
                score = f1
                # High bonus for approaching 90%
                if prec >= 0.85 and rec >= 0.85:
                    score *= 3.0
                elif prec >= 0.80 and rec >= 0.80:
                    score *= 2.5
                elif prec >= 0.75 and rec >= 0.75:
                    score *= 2.0
                elif prec >= 0.70 and rec >= 0.70:
                    score *= 1.5
                
                # Penalize imbalance
                if abs(prec - rec) > 0.15:
                    score *= 0.8
                    
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_precision = prec
                    best_recall = rec
        
        return best_threshold, best_precision, best_recall
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs[:, 1].cpu().detach().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        optimal_thresh, opt_prec, opt_rec = self.find_optimal_threshold(all_labels, np.array(all_probs))
        preds_thresh = (np.array(all_probs) >= optimal_thresh).astype(int)
        
        precision = precision_score(all_labels, preds_thresh, average='binary', zero_division=0)
        recall = recall_score(all_labels, preds_thresh, average='binary', zero_division=0)
        f1 = f1_score(all_labels, preds_thresh, average='binary', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, preds_thresh)
        
        return epoch_loss, epoch_acc, precision, recall, f1, balanced_acc, optimal_thresh
    
    def validate(self, val_loader, criterion, use_threshold=None):
        """Validate model with optional threshold"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        if use_threshold is None:
            optimal_thresh, opt_prec, opt_rec = self.find_optimal_threshold(all_labels, np.array(all_probs))
        else:
            optimal_thresh = use_threshold
        
        preds_thresh = (np.array(all_probs) >= optimal_thresh).astype(int)
        
        precision = precision_score(all_labels, preds_thresh, average='binary', zero_division=0)
        recall = recall_score(all_labels, preds_thresh, average='binary', zero_division=0)
        f1 = f1_score(all_labels, preds_thresh, average='binary', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, preds_thresh)
        
        return epoch_loss, epoch_acc, precision, recall, f1, balanced_acc, all_labels, preds_thresh, all_probs, optimal_thresh
    
    def train(self, train_loader, val_loader, num_epochs=40, lr=0.001, 
              patience=8, weight_decay=1e-3, use_focal_loss=False):
        """Training loop with precision focus and anti-overfitting"""
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=1, gamma=1.5, weight=self.class_weights)
            print("Using Focal Loss (gamma=1.5) with balanced class weighting")
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"Using Weighted CrossEntropy Loss (balanced weights): {self.class_weights}")
        
        fine_tune_params = []
        new_params = []
        for name, param in self.model.named_parameters():
            if 'feature_extractor' in name and param.requires_grad:
                fine_tune_params.append(param)
            else:
                new_params.append(param)
        
        optimizer = optim.Adam([
            {'params': fine_tune_params, 'lr': lr * 0.1},
            {'params': new_params, 'lr': lr}
        ], weight_decay=weight_decay)  # Higher weight_decay for regularization
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        patience_counter = 0
        target_met = False
        
        print(f"\n{'='*50}")
        print(f"Training {self.model_name}")
        print(f"Target: Precision >70%, Recall >70%, Minimize Overfitting")
        print(f"{'='*50}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc, train_prec, train_rec, train_f1, train_bal_acc, train_thresh = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_bal_acc, _, _, _, val_thresh = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_precisions.append(train_prec)
            self.train_recalls.append(train_rec)
            self.val_precisions.append(val_prec)
            self.val_recalls.append(val_rec)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)
            
            print(f"\nTrain Metrics (threshold={train_thresh:.3f}):")
            print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.2f}%) | Bal Acc: {train_bal_acc:.4f}")
            print(f"  Precision: {train_prec:.4f} ({train_prec*100:.2f}%) | Recall: {train_rec:.4f} ({train_rec*100:.2f}%) | F1: {train_f1:.4f}")
            
            print(f"\nVal Metrics (threshold={val_thresh:.3f}):")
            print(f"  Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | Bal Acc: {val_bal_acc:.4f}")
            print(f"  Precision: {val_prec:.4f} ({val_prec*100:.2f}%) | Recall: {val_rec:.4f} ({val_rec*100:.2f}%) | F1: {val_f1:.4f}")
            
            # Check overfitting
            prec_gap = train_prec - val_prec
            if prec_gap > 0.30:
                print(f"  ⚠ Overfitting! Train-Val Precision gap: {prec_gap:.2f}")
            
            # Model selection: STRONG preference for precision (fixes 40% issue)
            base_score = val_f1
            
            # Score calculation with precision priority
            if val_prec >= 0.70 and val_rec >= 0.70:
                # Both targets met - best case
                score = base_score * 3.0
                if val_prec >= 0.75:
                    score *= 1.3  # Extra precision bonus
                if val_acc >= 0.80:
                    score *= 1.2
            elif val_prec >= 0.70 and val_rec >= 0.65:
                # Precision met, recall close
                score = base_score * 2.5
            elif val_prec >= 0.65 and val_rec >= 0.70:
                # Precision close, recall met
                score = base_score * 2.0  # Lower than above (precision priority)
            elif val_prec >= 0.60 and val_rec >= 0.70:
                # Precision improving, recall good
                score = base_score * 1.8
            elif val_prec >= 0.50 and val_rec >= 0.65:
                score = base_score * 1.3
            else:
                score = base_score
            
            score += val_acc * 0.3
            score += val_bal_acc * 0.2
            
            # Penalize if overfitting is severe
            if prec_gap > 0.40:
                score *= 0.7  # Reduce score if severe overfitting
            
            if score > self.best_val_balanced_f1:
                self.best_val_balanced_f1 = score
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_threshold = val_thresh
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if val_prec >= 0.70 and val_rec >= 0.70:
                    target_met = True
                    print(f"\n✓✓ NEW BEST MODEL! TARGETS MET!")
                else:
                    print(f"\n✓ NEW BEST MODEL!")
                    
                print(f"  Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Val Bal Acc: {val_bal_acc:.4f}")
                print(f"  Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | Threshold: {val_thresh:.3f}")
                
                if val_prec >= 0.70 and val_rec >= 0.70:
                    print(f"  ✓ TARGET ACHIEVED: Both precision and recall > 70%!")
                elif val_prec >= 0.65:
                    print(f"  → Precision improving: {val_prec:.2%}")
            else:
                patience_counter += 1
            
            # Early stopping
            if target_met and patience_counter >= 5:
                print(f"\nTargets met! Early stopping at epoch {epoch+1}")
                break
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            print("-" * 50)
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest validation F1: {self.best_val_f1:.4f}")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"Optimal threshold: {self.best_threshold:.3f}")
        
        return self.model
    
    def evaluate(self, test_loader, threshold=None):
        """Comprehensive evaluation on test set"""
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        test_threshold = threshold if threshold is not None else self.best_threshold
        
        test_loss, test_acc, test_prec, test_rec, test_f1, test_bal_acc, y_true, y_pred, y_probs, _ = self.validate(test_loader, criterion, use_threshold=test_threshold)
        
        try:
            auc = roc_auc_score(y_true, y_probs)
        except:
            auc = 0.0
        
        y_pred_thresh = (np.array(y_probs) >= test_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1,
            'auc_roc': auc,
            'specificity': specificity,
            'confusion_matrix': cm,
            'threshold': test_threshold,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
        
        print(f"\n{'='*50}")
        print(f"Test Set Evaluation - {self.model_name}")
        print(f"Using threshold: {test_threshold:.3f}")
        print(f"{'='*50}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Balanced Accuracy: {test_bal_acc:.4f} ({test_bal_acc*100:.2f}%)")
        print(f"\n--- Classification Metrics ---")
        print(f"Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
        print(f"Recall: {test_rec:.4f} ({test_rec*100:.2f}%)")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\n--- Confusion Matrix ---")
        print(f"                Predicted")
        print(f"              Melanoma  Non-Melanoma")
        print(f"Actual Melanoma     {tp:4d}        {fn:4d}")
        print(f"      Non-Melanoma  {fp:4d}        {tn:4d}")
        print(f"\nBreakdown:")
        print(f"  True Positives (TP):  {tp:4d} - Correctly identified melanomas")
        print(f"  False Positives (FP): {fp:4d} - Non-melanomas misclassified as melanoma")
        print(f"  True Negatives (TN):  {tn:4d} - Correctly identified non-melanomas")
        print(f"  False Negatives (FN): {fn:4d} - MISSED melanomas")
        
        if test_rec >= 0.70 and test_prec >= 0.70:
            print(f"\n✓✓ TARGET ACHIEVED! Both precision and recall > 70%!")
            print(f"   Recall: {test_rec*100:.1f}% | Precision: {test_prec*100:.1f}% | Accuracy: {test_acc*100:.1f}%")
            if test_acc >= 0.80:
                print(f"   ✓ Excellent accuracy too!")
        elif test_rec >= 0.65 and test_prec >= 0.65:
            print(f"\n✓ GOOD: Both metrics > 65% (Recall: {test_rec*100:.1f}%, Precision: {test_prec*100:.1f}%)")
        else:
            print(f"\n⚠ NEEDS IMPROVEMENT: One or both metrics < 70%")
            print(f"   Recall: {test_rec*100:.1f}% | Precision: {test_prec*100:.1f}% | Accuracy: {test_acc*100:.1f}%")
        
        print(f"{'='*50}\n")
        
        return metrics
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.train_accs, label='Train Acc')
        axes[0, 1].plot(self.val_accs, label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=0.70, color='g', linestyle='--', alpha=0.5, label='Target (70%)')
        
        axes[0, 2].plot(self.train_precisions, label='Train Precision')
        axes[0, 2].plot(self.val_precisions, label='Val Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Training and Validation Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[0, 2].axhline(y=0.70, color='g', linestyle='--', alpha=0.5, label='Target (70%)')
        
        axes[1, 0].plot(self.train_recalls, label='Train Recall')
        axes[1, 0].plot(self.val_recalls, label='Val Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Training and Validation Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0.70, color='g', linestyle='--', alpha=0.5, label='Target (70%)')
        
        axes[1, 1].plot(self.train_f1s, label='Train F1')
        axes[1, 1].plot(self.val_f1s, label='Val F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Training and Validation F1-Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(self.val_recalls, self.val_precisions, 'o-', label='Val')
        axes[1, 2].plot(self.train_recalls, self.train_precisions, 's--', label='Train', alpha=0.7)
        axes[1, 2].set_xlabel('Recall')
        axes[1, 2].set_ylabel('Precision')
        axes[1, 2].set_title('Precision-Recall Tradeoff')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        axes[1, 2].axhline(y=0.70, color='g', linestyle='--', alpha=0.3)
        axes[1, 2].axvline(x=0.70, color='g', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training history saved to {save_path}")