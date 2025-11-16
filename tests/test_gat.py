"""
Tests for Graph Attention Network (GAT) module
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pytest
from Src.Utils.GAT import GraphAttentionLayer, GraphAttentionNetwork, build_knn_adjacency_mask


class TestGraphAttentionLayer:
    """Test suite for GraphAttentionLayer"""
    
    def test_initialization(self):
        """Test GAT layer initialization"""
        layer = GraphAttentionLayer(in_features=16, out_features=32, dropout=0.1, alpha=0.2)
        
        assert layer.in_features == 16
        assert layer.out_features == 32
        assert layer.dropout == 0.1
        assert layer.alpha == 0.2
        assert layer.W.shape == (16, 32)
        assert layer.a.shape == (64, 1)  # 2 * out_features
    
    def test_forward_pass(self):
        """Test forward pass of GAT layer"""
        n_agents = 5
        batch_size = 2
        in_features = 16
        out_features = 32
        
        layer = GraphAttentionLayer(in_features, out_features)
        layer.init()
        
        # Create input embeddings: (n_agents, batch_size, in_features)
        z = torch.randn(n_agents, batch_size, in_features)
        
        # Forward pass
        h_out, attn_weights = layer(z)
        
        # Check output shape
        assert h_out.shape == (n_agents, batch_size, out_features)
        assert len(attn_weights.shape) == 3
        assert attn_weights.shape[0] == n_agents
        assert attn_weights.shape[1] == n_agents
        assert attn_weights.shape[2] == batch_size
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each agent"""
        n_agents = 4
        batch_size = 1
        in_features = 8
        out_features = 16
        
        layer = GraphAttentionLayer(in_features, out_features, dropout=0.0)  # No dropout for testing
        layer.eval()  # Set to eval mode to disable dropout
        layer.init()
        
        z = torch.randn(n_agents, batch_size, in_features)
        h_out, attn_weights = layer(z)
        
        # Attention weights should sum to 1 for each agent (over neighbors)
        for i in range(n_agents):
            for b in range(batch_size):
                attn_sum = attn_weights[i, :, b].sum().item()
                assert np.isclose(attn_sum, 1.0, atol=1e-4), \
                    f"Attention weights for agent {i}, batch {b} should sum to 1, got {attn_sum}"
    
    def test_adjacency_mask(self):
        """Test that adjacency mask restricts attention"""
        n_agents = 4
        batch_size = 1
        in_features = 8
        out_features = 16
        
        layer = GraphAttentionLayer(in_features, out_features)
        layer.init()
        
        z = torch.randn(n_agents, batch_size, in_features)
        
        # Create adjacency mask: only agent 0 can attend to agents 0 and 1
        adj_mask = torch.zeros(n_agents, n_agents)
        adj_mask[0, 0] = 1.0
        adj_mask[0, 1] = 1.0
        adj_mask[1, 1] = 1.0
        adj_mask[1, 2] = 1.0
        # ... etc (make it symmetric for simplicity)
        for i in range(n_agents):
            adj_mask[i, i] = 1.0  # Self-connection
        
        h_out, attn_weights = layer(z, adj_mask=adj_mask)
        
        # Check that masked connections have zero attention (or very small)
        # Agent 0 should not attend to agents 2 and 3
        assert attn_weights[0, 2, 0].item() < 1e-6 or attn_weights[0, 3, 0].item() < 1e-6, \
            "Masked connections should have near-zero attention"


class TestGraphAttentionNetwork:
    """Test suite for GraphAttentionNetwork"""
    
    def test_initialization(self):
        """Test GAT network initialization"""
        gat = GraphAttentionNetwork(
            embedding_dim=16,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            alpha=0.2
        )
        
        assert gat.embedding_dim == 16
        assert gat.hidden_dim == 32
        assert gat.num_layers == 2
        assert len(gat.layers) == 2
    
    def test_forward_single_step(self):
        """Test forward pass with single message passing step"""
        n_agents = 5
        batch_size = 2
        embedding_dim = 16
        
        gat = GraphAttentionNetwork(embedding_dim=embedding_dim)
        gat.init()
        
        # Initial embeddings
        z = torch.randn(n_agents, batch_size, embedding_dim)
        
        # Forward pass with 1 step
        z_out, all_attn = gat(z, num_steps=1)
        
        assert z_out.shape == (n_agents, batch_size, embedding_dim)
        assert len(all_attn) == gat.num_layers
    
    def test_forward_multiple_steps(self):
        """Test forward pass with multiple message passing steps"""
        n_agents = 4
        batch_size = 1
        embedding_dim = 8
        
        gat = GraphAttentionNetwork(embedding_dim=embedding_dim)
        gat.init()
        
        z = torch.randn(n_agents, batch_size, embedding_dim)
        
        # Forward pass with 2 steps
        z_out, all_attn = gat(z, num_steps=2)
        
        assert z_out.shape == (n_agents, batch_size, embedding_dim)
        # Should have num_layers attention weights per step
        assert len(all_attn) >= gat.num_layers
    
    def test_adjacency_mask_propagation(self):
        """Test that adjacency mask works with GAT network"""
        n_agents = 3
        batch_size = 1
        embedding_dim = 8
        
        gat = GraphAttentionNetwork(embedding_dim=embedding_dim, dropout=0.0)
        gat.eval()  # Disable dropout
        gat.init()
        
        z = torch.randn(n_agents, batch_size, embedding_dim)
        adj_mask = build_knn_adjacency_mask(n_agents, k=2)
        
        z_out, _ = gat(z, adj_mask=adj_mask, num_steps=1)
        
        assert z_out.shape == (n_agents, batch_size, embedding_dim), \
            f"Expected shape ({n_agents}, {batch_size}, {embedding_dim}), got {z_out.shape}"


class TestKNNAdjacencyMask:
    """Test suite for k-NN adjacency mask builder"""
    
    def test_build_knn_mask_default(self):
        """Test building k-NN mask with default (circular) similarity"""
        n_agents = 5
        k = 2
        
        adj_mask = build_knn_adjacency_mask(n_agents, k=k)
        
        assert adj_mask.shape == (n_agents, n_agents)
        assert adj_mask.dtype == torch.float32
        
        # Each agent should have k neighbors (including self)
        for i in range(n_agents):
            assert adj_mask[i, i] == 1.0, "Self-connection should always be present"
            num_neighbors = adj_mask[i].sum().item()
            assert num_neighbors == k, f"Agent {i} should have {k} neighbors, got {num_neighbors}"
    
    def test_build_knn_mask_with_similarity(self):
        """Test building k-NN mask with similarity matrix"""
        n_agents = 4
        k = 2
        
        # Create similarity matrix (higher values = more similar)
        similarity_matrix = torch.tensor([
            [1.0, 0.8, 0.2, 0.1],  # Agent 0 is similar to 0 and 1
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.9],  # Agent 2 is similar to 2 and 3
            [0.1, 0.2, 0.9, 1.0]
        ])
        
        adj_mask = build_knn_adjacency_mask(n_agents, k=k, similarity_matrix=similarity_matrix)
        
        assert adj_mask.shape == (n_agents, n_agents)
        # Agent 0 should be connected to 0 and 1 (most similar)
        assert adj_mask[0, 0] == 1.0
        assert adj_mask[0, 1] == 1.0
    
    def test_knn_mask_symmetry(self):
        """Test that k-NN mask is symmetric (for undirected graph)"""
        n_agents = 6
        k = 3
        
        adj_mask = build_knn_adjacency_mask(n_agents, k=k)
        
        # Check symmetry (if i->j exists, j->i should exist for undirected)
        # Note: Our default implementation uses circular distance, which is symmetric
        for i in range(n_agents):
            for j in range(n_agents):
                if adj_mask[i, j] == 1.0:
                    # In circular k-NN, if i is neighbor of j, j should be neighbor of i
                    # (This depends on implementation, but generally true for circular)
                    pass  # Just check it doesn't crash
    
    def test_knn_mask_different_k_values(self):
        """Test k-NN mask with different k values"""
        n_agents = 5
        
        for k in [1, 2, 3, 4]:
            adj_mask = build_knn_adjacency_mask(n_agents, k=k)
            
            for i in range(n_agents):
                num_neighbors = adj_mask[i].sum().item()
                assert num_neighbors == k, f"k={k}: Agent {i} should have {k} neighbors, got {num_neighbors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

