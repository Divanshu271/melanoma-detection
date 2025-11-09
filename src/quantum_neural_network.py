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
        
        # Classical features
        features = self.feature_extractor(x)
        features = features.view(batch_size, -1)
        
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
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds.extend(outputs.argmax(1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_f1 = f1_score(labels, preds, average='macro')
            
            # Validation
            self.eval()
            val_loss = 0
            preds, labels = [], []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    outputs = self(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    preds.extend(outputs.argmax(1).cpu().numpy())
                    labels.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_f1 = f1_score(labels, preds, average='macro')
            
            # Update scheduler
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = self.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # Load best model
        if best_state is not None:
            self.load_state_dict(best_state)
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        device = next(self.parameters()).device
        self.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                outputs = self(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(targets.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='macro'),
            'auc_roc': roc_auc_score(all_labels, all_probs)
        }
        
        return metrics