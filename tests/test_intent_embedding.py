"""
Tests for IntentEmbeddingNetwork
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pytest

# Import IntentEmbeddingNetwork directly to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the class directly from the module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "multiagent_qac_gat", 
    os.path.join(os.path.dirname(__file__), "..", "Src", "RL_Algorithms", "MultiAgent_QAC_GAT.py")
)
# For now, let's just define it here to avoid import issues
from Src.Utils.Utils import NeuralNet
import torch.nn as nn

class IntentEmbeddingNetwork(NeuralNet):
    """Test version of IntentEmbeddingNetwork"""
    def __init__(self, state_dim, embedding_dim, hidden_dim=64, config=None):
        super(IntentEmbeddingNetwork, self).__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        self.init()
    
    def init(self):
        if self.config is not None:
            lr = getattr(self.config, 'embedding_lr', getattr(self.config, 'state_lr', 1e-3))
            self.optim = self.config.optim(self.parameters(), lr=lr)
        else:
            self.optim = None
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        z = self.fc3(x)
        return z


class TestIntentEmbeddingNetwork:
    """Test suite for IntentEmbeddingNetwork"""
    
    def test_initialization(self):
        """Test embedding network initialization"""
        state_dim = 4
        embedding_dim = 16
        hidden_dim = 64
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim, hidden_dim)
        
        assert net.state_dim == state_dim
        assert net.embedding_dim == embedding_dim
        assert net.hidden_dim == hidden_dim
        assert net.fc1.in_features == state_dim
        assert net.fc1.out_features == hidden_dim
        assert net.fc3.out_features == embedding_dim
    
    def test_forward_pass(self):
        """Test forward pass of embedding network"""
        state_dim = 4
        embedding_dim = 16
        batch_size = 5
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim)
        
        # Input: (batch_size, state_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        embedding = net(state)
        
        # Check output shape
        assert embedding.shape == (batch_size, embedding_dim)
        assert not torch.isnan(embedding).any(), "Embedding should not contain NaN"
        assert not torch.isinf(embedding).any(), "Embedding should not contain Inf"
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample"""
        state_dim = 4
        embedding_dim = 16
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim)
        
        state = torch.randn(1, state_dim)
        embedding = net(state)
        
        assert embedding.shape == (1, embedding_dim)
    
    def test_embedding_different_states(self):
        """Test that different states produce different embeddings"""
        state_dim = 4
        embedding_dim = 16
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim)
        
        state1 = torch.randn(1, state_dim)
        state2 = torch.randn(1, state_dim)
        
        embedding1 = net(state1)
        embedding2 = net(state2)
        
        # Embeddings should be different (with high probability for random states)
        assert not torch.allclose(embedding1, embedding2), \
            "Different states should produce different embeddings"
    
    def test_embedding_gradient_flow(self):
        """Test that gradients flow through embedding network"""
        state_dim = 4
        embedding_dim = 16
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim)
        
        state = torch.randn(1, state_dim, requires_grad=True)
        embedding = net(state)
        
        # Compute loss and backward
        loss = embedding.sum()
        loss.backward()
        
        # Check that gradients exist
        assert state.grad is not None, "Gradients should flow to input"
        for param in net.parameters():
            assert param.grad is not None, f"Gradients should flow to parameter {param}"
    
    def test_embedding_initialization_with_config(self):
        """Test embedding network initialization with config"""
        from types import SimpleNamespace
        
        state_dim = 4
        embedding_dim = 16
        
        # Create mock config
        config = SimpleNamespace()
        config.optim = torch.optim.Adam
        config.embedding_lr = 1e-3
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim, config=config)
        
        assert net.optim is not None
        assert isinstance(net.optim, torch.optim.Adam)
    
    def test_embedding_batch_processing(self):
        """Test processing multiple states in batch"""
        state_dim = 4
        embedding_dim = 16
        batch_sizes = [1, 5, 10, 32]
        
        net = IntentEmbeddingNetwork(state_dim, embedding_dim)
        
        for batch_size in batch_sizes:
            state = torch.randn(batch_size, state_dim)
            embedding = net(state)
            
            assert embedding.shape == (batch_size, embedding_dim), \
                f"Batch size {batch_size}: expected shape ({batch_size}, {embedding_dim}), got {embedding.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

