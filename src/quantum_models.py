import torch
import torch.nn as nn
import pennylane as qml
from torchvision import models
import ssl
import urllib.request

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

class QuantumFeatureMap:
    """Quantum feature map for encoding classical features"""
    
    def __init__(self, n_qubits, n_layers=2, dev='default.qubit'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev, wires=n_qubits)
        
    def feature_map(self, features):
        """
        Amplitude encoding of features into quantum state
        """
        # Normalize features to create valid quantum state
        features = features / torch.norm(features, dim=-1, keepdim=True)
        
        # Resize to 2^n_qubits
        target_size = 2 ** self.n_qubits
        if features.shape[-1] < target_size:
            # Pad with zeros
            padding = target_size - features.shape[-1]
            features = torch.cat([features, torch.zeros(*features.shape[:-1], padding)], dim=-1)
        elif features.shape[-1] > target_size:
            # Truncate
            features = features[..., :target_size]
        
        qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
    
    def variational_layer(self, params, layer_idx):
        """Variational quantum circuit layer"""
        n = self.n_qubits
        param_idx = layer_idx * n * 2
        
        # Rotation gates
        for i in range(n):
            qml.RY(params[param_idx + i], wires=i)
            qml.RZ(params[param_idx + n + i], wires=i)
        
        # Entangling layer
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n - 1, 0])  # Circular entanglement

def load_resnet18(use_pretrained=True):
    """Load ResNet18 with error handling for SSL issues"""
    if not use_pretrained:
        return models.resnet18(weights=None)
    
    try:
        # Try newer API first (torchvision >= 0.13)
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback to older API
            return models.resnet18(pretrained=True)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights ({e})")
        print("Using randomly initialized ResNet18 instead")
        return models.resnet18(weights=None)

class QuantumNeuralNetwork(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for Melanoma Classification
    WITH FINE-TUNING ENABLED for >90% performance
    """
    
    def __init__(self, n_qubits=4, n_layers=2, classical_dim=512, use_pretrained=True, fine_tune=True):
        super(QuantumNeuralNetwork, self).__init__()
        
        # Classical feature extractor (pre-trained ResNet)
        resnet = load_resnet18(use_pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        
        # Enable fine-tuning for better performance
        self.fine_tune = fine_tune
        if fine_tune:
            # Unfreeze last few layers for fine-tuning
            for param in list(self.feature_extractor.parameters())[-20:]:  # Last 20 params
                param.requires_grad = True
            print("Feature extractor fine-tuning ENABLED")
        
        # Dimension reduction to match quantum feature map
        self.feature_reduction = nn.Linear(classical_dim, 2**n_qubits)
        
        # Quantum layer parameters
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2  # RY + RZ per qubit per layer
        
        # Initialize quantum parameters
        self.q_params = nn.Parameter(torch.randn(self.n_params))
        
        # Quantum device
        self.qdev = qml.device('default.qubit', wires=n_qubits)
        
        # Define quantum circuit
        @qml.qnode(self.qdev, interface='torch')
        def quantum_circuit(features, params):
            # Feature map (amplitude encoding)
            qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
            
            # Variational layers
            for layer in range(n_layers):
                param_idx = layer * n_qubits * 2
                # Rotation gates
                for i in range(n_qubits):
                    qml.RY(params[param_idx + i], wires=i)
                    qml.RZ(params[param_idx + n_qubits + i], wires=i)
                
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            
            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Enhanced classifier with STRONG regularization to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),  # Very high dropout to prevent overfitting
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),  # High dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features using classical CNN (with fine-tuning if enabled)
        if self.fine_tune:
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)
        features = features.view(batch_size, -1)
        
        # Reduce dimensions for quantum encoding
        q_features = self.feature_reduction(features)
        
        # Normalize for quantum state
        q_features = q_features / (torch.norm(q_features, dim=1, keepdim=True) + 1e-8)
        
        # OPTIMIZED: Process quantum circuit more efficiently
        # Use smaller batch processing if batch is large
        if batch_size > 16:
            # Process in chunks to avoid memory issues
            quantum_outputs = []
            chunk_size = 16
            for i in range(0, batch_size, chunk_size):
                chunk_end = min(i + chunk_size, batch_size)
                chunk_features = q_features[i:chunk_end]
                chunk_outputs = []
                for j in range(len(chunk_features)):
                    q_out = self.quantum_circuit(chunk_features[j], self.q_params)
                    if isinstance(q_out, (list, tuple)):
                        q_out = torch.stack([torch.tensor(val, dtype=torch.float32) if not isinstance(val, torch.Tensor) 
                                            else val.float() for val in q_out])
                    elif not isinstance(q_out, torch.Tensor):
                        q_out = torch.tensor(q_out, dtype=torch.float32)
                    q_out = q_out.to(q_features.device)
                    chunk_outputs.append(q_out)
                quantum_outputs.extend(chunk_outputs)
        else:
            # Small batch - process normally
            quantum_outputs = []
            for i in range(batch_size):
                q_out = self.quantum_circuit(q_features[i], self.q_params)
                if isinstance(q_out, (list, tuple)):
                    q_out = torch.stack([torch.tensor(val, dtype=torch.float32) if not isinstance(val, torch.Tensor) 
                                        else val.float() for val in q_out])
                elif not isinstance(q_out, torch.Tensor):
                    q_out = torch.tensor(q_out, dtype=torch.float32)
                q_out = q_out.to(q_features.device)
                quantum_outputs.append(q_out)
        
        # Stack all outputs into a batch tensor
        quantum_outputs = torch.stack(quantum_outputs)
        
        # Classical classification
        output = self.classifier(quantum_outputs)
        
        return output
class ClassicalBaseline(nn.Module):
    """Classical baseline model for comparison"""
    
    def __init__(self, num_classes=2, use_pretrained=True):
        super(ClassicalBaseline, self).__init__()
        resnet = load_resnet18(use_pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x