"""
Optimized Quantum Neural Network with proper batching and vectorization
Uses PennyLane's TorchLayer for efficient quantum circuit execution
"""
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from src.metrics_utils import classification_metrics
import warnings
warnings.filterwarnings('ignore')

class QuantumLayer(nn.Module):
    """
    Quantum layer using PennyLane's TorchLayer
    Properly handles batching and gradients
    """
    
    def __init__(self, n_qubits=6, n_layers=2, input_size=512):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_size = input_size
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Create QNode
        self.q_layer = qml.QNode(
            self.quantum_circuit,
            self.dev,
            interface="torch",
            diff_method="backprop"
        )
        
        # Convert to TorchLayer for batching
        self.qlayer = qml.qnn.TorchLayer(
            self.q_layer,
            weight_shapes={
                "weights": (n_layers, n_qubits, 3)  # 3 rotation parameters per qubit
            }
        )
        
        # Linear layer for dimensionality reduction
        self.linear = nn.Linear(input_size, n_qubits)
        
    def quantum_circuit(self, inputs, weights):
        """
        Quantum circuit with angle embedding and parameterized gates
        """
        # Angle embedding of input features
        AngleEmbedding(inputs, wires=range(self.n_qubits))
        
        # Parameterized quantum circuit
        StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        
        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward pass with batching"""
        # Reduce dimensionality
        x = self.linear(x)
        
        # Normalize for quantum state
        x = x / torch.norm(x, dim=1, keepdim=True)
        
        # Apply quantum layer (handles batching internally)
        return self.qlayer(x)

class OptimizedQNN(nn.Module):
    """
    Optimized Quantum Neural Network with:
    1. Proper batching using TorchLayer
    2. Improved classical-quantum hybrid architecture
    3. Strong regularization and residual connections
    """
    
    def __init__(self, n_qubits=6, n_layers=2, classical_dim=512,
                 dropout_rate=0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.classical_dim = classical_dim
        
        # Classical feature extractor
        self.feature_extractor = self._create_feature_extractor()
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            input_size=classical_dim
        )
        
        # Enhanced classical layers with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits + classical_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(64, 2)
        )
        # Best decision threshold (for positive class) chosen on validation
        self.best_threshold = 0.5
    # Threshold tuning behavior: 'balanced_min' (default) or 'recall' or 'precision' or 'f1'
        # If 'recall', we maximize recall subject to min_precision.
        # If 'precision', we maximize precision subject to min_recall.
        # If 'f1', we maximize F1 directly (no additional constraint used).
    # 'balanced_min' maximizes min(precision, recall) to get balanced performance
        self.threshold_optimize = 'balanced_min'
        self.min_precision = 0.5
        self.min_recall = 0.0
        # Optional use of focal loss
        self.use_focal = False
        self.focal_gamma = 2.0
        # Optional projection layer to map precomputed embeddings -> expected classical_dim
        self.embedding_proj = None
    
    def _create_feature_extractor(self):
        """Create ResNet feature extractor"""
        from torchvision.models import resnet18, ResNet18_Weights
        
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final layers
        modules = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
        # Freeze early layers
        for param in list(feature_extractor.parameters())[:-20]:
            param.requires_grad = False
            
        return feature_extractor
    
    def forward(self, x):
        """Forward pass with classical-quantum hybrid processing"""
        batch_size = x.size(0)

        # Support both raw image tensors and precomputed embeddings (pseudo-images).
        # - If input is a 4D tensor with 3 channels (B,3,H,W) assume image and run feature_extractor.
        # - If input is a 3D tensor shaped (B,1,embedding_dim) or (B, embedding_dim) treat as embeddings
        #   produced elsewhere (ResNet embeddings). This avoids passing 1-channel pseudo-images
        #   through the ResNet stem which expects 3 channels.
        if x.dim() == 4 and x.size(1) == 3:
            # Classical image path
            features = self.feature_extractor(x)
            features = features.view(batch_size, -1)
        else:
            # Embedding / pseudo-image path: try to collapse to (batch, classical_dim)
            if x.dim() == 3 and x.size(1) == 1:
                # shape (B,1,embedding_dim)
                features = x.squeeze(1)
            else:
                # Any other shape: flatten trailing dims
                features = x.view(batch_size, -1)

            # If embeddings are smaller than expected, project to expected classical_dim
            if features.size(1) != self.classical_dim:
                in_dim = features.size(1)
                # Create or re-create persistent projection layer if shape differs
                if self.embedding_proj is None or getattr(self.embedding_proj, 'in_features', None) != in_dim:
                    self.embedding_proj = nn.Linear(in_dim, self.classical_dim).to(features.device)
                # Ensure projection lives on the same device
                if next(self.embedding_proj.parameters()).device != features.device:
                    self.embedding_proj = self.embedding_proj.to(features.device)
                features = self.embedding_proj(features.float())
        # Quantum processing
        quantum_out = self.quantum_layer(features)

        # Combine classical and quantum features
        combined = torch.cat([quantum_out, features], dim=1)

        # Classification
        logits = self.classifier(combined)
        return logits
    
    def train_fold(self, train_loader, val_loader, epochs=50,
                   lr=0.001, fold=None, class_weights=None):
        """Train model for one cross-validation fold"""
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Use class weights for balanced loss if provided
        if self.use_focal:
            # implement focal loss with optional class weights
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, weight=None):
                    super().__init__()
                    self.gamma = gamma
                    self.weight = weight

                def forward(self, inputs, targets):
                    # inputs: logits (N, C)
                    probs = torch.softmax(inputs, dim=1)
                    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(1)).float()
                    pt = (probs * targets_one_hot).sum(dim=1)
                    log_pt = torch.log(pt + 1e-12)
                    loss = -((1 - pt) ** self.gamma) * log_pt
                    if self.weight is not None:
                        wt = self.weight[targets].to(inputs.device)
                        loss = loss * wt
                    return loss.mean()

            weight = class_weights.to(device) if class_weights is not None else None
            criterion = FocalLoss(gamma=self.focal_gamma, weight=weight)
        else:
            if class_weights is not None:
                class_weights = class_weights.to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0
            preds, labels = [], []
            
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds.extend(outputs.argmax(1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            # Use positive-class (binary) F1 for training/validation logging
            train_f1 = f1_score(labels, preds, average='binary', zero_division=0)
            
            # Validation
            self.eval()
            val_loss = 0
            preds, labels = [], []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    outputs = self(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    preds.extend(outputs.argmax(1).cpu().numpy())
                    labels.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_f1 = f1_score(labels, preds, average='binary', zero_division=0)
            
            # Update scheduler
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = self.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, F1 (pos): {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, F1 (pos): {val_f1:.4f}")
        
        # Load best model
        if best_state is not None:
            self.load_state_dict(best_state)
        
        # After training, compute validation probabilities and find threshold to
        # maximize recall subject to a minimum precision (helps increase recall)
        try:
            if val_loader is not None:
                self.best_threshold = self._find_threshold_on_loader(
                    val_loader,
                    min_precision=self.min_precision,
                    min_recall=self.min_recall,
                    optimize_for=self.threshold_optimize
                )
                print(f"Selected validation threshold for positive class ({self.threshold_optimize}): {self.best_threshold:.3f}")
        except Exception:
            # fallback
            self.best_threshold = 0.5
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        device = next(self.parameters()).device
        self.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                outputs = self(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(targets.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Instead of using argmax preds, apply the validation-tuned threshold if available
        try:
            probs_arr = np.array(all_probs)
            pred_by_thresh = (probs_arr >= self.best_threshold).astype(int)
            metrics = classification_metrics(all_labels, pred_by_thresh, probs_arr, pos_label=1)
        except Exception:
            metrics = classification_metrics(all_labels, all_preds, np.array(all_probs), pos_label=1)

        return metrics

    def _find_threshold_on_loader(self, loader, min_precision=0.5, min_recall=0.0, optimize_for='recall'):
        """Sweep thresholds on a validation loader and pick the best threshold.

        Parameters
        - loader: validation DataLoader that yields (images, targets)
        - min_precision: minimum precision constraint when optimizing for recall
        - min_recall: minimum recall constraint when optimizing for precision
        - optimize_for: one of {'recall', 'precision', 'f1'}

        Behavior:
        - 'recall': choose threshold that maximizes recall while keeping precision >= min_precision.
        - 'precision': choose threshold that maximizes precision while keeping recall >= min_recall.
        - 'f1': choose threshold that maximizes F1 (no extra constraint).

        If no threshold satisfies the constraint, the function falls back to the threshold
        that maximizes the requested objective without the constraint (or 0.5 if something fails).
        Returns a float threshold in [0,1].
        """
        device = next(self.parameters()).device
        self.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                outputs = self(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        best_thresh = 0.5

        # candidate arrays
        thresholds = np.linspace(0.0, 1.0, 101)
        from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

        scores = []
        for t in thresholds:
            y_pred = (all_probs >= t).astype(int)
            prec = precision_score(all_labels, y_pred, zero_division=0)
            rec = recall_score(all_labels, y_pred, zero_division=0)
            f1 = f1_score(all_labels, y_pred, zero_division=0)
            bal = balanced_accuracy_score(all_labels, y_pred)
            scores.append((float(t), float(prec), float(rec), float(f1), float(bal)))

        # Selection modes: 'recall', 'precision', 'f1', 'balanced_min'
        selected = None
        if optimize_for == 'recall':
            # keep thresholds that meet precision constraint
            valid = [s for s in scores if s[1] >= float(min_precision)]
            if len(valid) > 0:
                # maximize recall, tie-breaker: higher precision then higher f1
                valid.sort(key=lambda x: (x[2], x[1], x[3]), reverse=True)
                selected = valid[0]
            else:
                # fallback: maximize recall regardless of precision
                scores.sort(key=lambda x: (x[2], x[1], x[3]), reverse=True)
                selected = scores[0]

        elif optimize_for == 'precision':
            valid = [s for s in scores if s[2] >= float(min_recall)]
            if len(valid) > 0:
                # maximize precision, tie-breaker: higher recall then higher f1
                valid.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
                selected = valid[0]
            else:
                # fallback: maximize precision regardless of recall
                scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
                selected = scores[0]

        elif optimize_for == 'balanced_min':
            # choose threshold that maximizes min(precision, recall)
            scores.sort(key=lambda x: (min(x[1], x[2]), x[4], x[3]), reverse=True)
            selected = scores[0]

        else:  # f1
            scores.sort(key=lambda x: x[3], reverse=True)
            selected = scores[0]

        try:
            best_thresh = float(selected[0])
        except Exception:
            best_thresh = 0.5

        return float(best_thresh)