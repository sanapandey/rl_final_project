"""
Tests for MultiAgent_QAC_GAT algorithm
Note: These tests require a mock config and environment setup
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pytest
from types import SimpleNamespace
from Src.RL_Algorithms.MultiAgent_QAC_GAT import MultiAgent_QAC_GAT, IntentEmbeddingNetwork
from Environments.JobShop.Multi_Agent_JobShop_py import Multi_Agent_JobShop_py


def create_mock_config(n_agents=3, n_jobs=50):
    """Create a mock config for testing"""
    # Create environment first
    env = Multi_Agent_JobShop_py(n_machines=n_agents, n_jobs=n_jobs, max_steps=100)
    
    config = SimpleNamespace()
    config.env = env
    config.env_name = 'Multi_Agent_JobShop_py'
    config.n_agents = n_agents
    config.embedding_dim = 16
    config.gat_k = 2
    config.gat_train_steps = 2
    config.gat_eval_steps = 1
    config.contrastive_weight = 0.1
    config.embedding_hidden_dim = 64
    config.gat_hidden_dim = 16
    config.gat_dropout = 0.1
    config.gat_alpha = 0.2
    config.gat_lr = 1e-3
    config.embedding_lr = 1e-3
    config.state_lr = 1e-3
    config.actor_lr = 1e-2
    config.critic_lr = 1e-2
    config.gamma = 0.99
    config.optim = torch.optim.Adam
    config.hiddenLayerSize = 32
    config.hiddenActorLayerSize = 16
    config.gauss_variance = 1.0
    config.actor_output_layer = 'tanh'
    config.actor_scaling_factor_mean = 1.0
    config.deepActor = False
    config.fourier_order = 0
    config.fourier_coupled = False
    config.save_model = False
    config.paths = {'checkpoint': '/tmp/test_checkpoints/'}
    
    return config


class TestMultiAgentQACGAT:
    """Test suite for MultiAgent_QAC_GAT"""
    
    def test_initialization(self):
        """Test algorithm initialization"""
        config = create_mock_config(n_agents=3)
        
        agent = MultiAgent_QAC_GAT(config)
        
        assert agent.n_agents == 3
        assert agent.embedding_dim == 16
        assert agent.gat_k == 2
        assert len(agent.embedding_networks) == 3
        assert len(agent.actors) == 3
        assert len(agent.critics) == 3
        assert agent.gat is not None
    
    def test_extract_intent_embeddings(self):
        """Test intent embedding extraction"""
        config = create_mock_config(n_agents=2)
        agent = MultiAgent_QAC_GAT(config)
        
        # Create mock observations (per-agent)
        observations = [
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32),  # Agent 0
            np.array([2.0, 0.3, 20.0, 15.0], dtype=np.float32)  # Agent 1
        ]
        
        embeddings = agent._extract_intent_embeddings(observations)
        
        # Check shape: (n_agents, 1, embedding_dim)
        assert embeddings.shape == (2, 1, 16)
        assert not torch.isnan(embeddings).any()
    
    def test_communicate_embeddings(self):
        """Test embedding communication through GAT"""
        config = create_mock_config(n_agents=3)
        agent = MultiAgent_QAC_GAT(config)
        
        # Create initial embeddings
        embeddings = torch.randn(3, 1, 16)
        
        # Communicate
        comm_embeddings = agent._communicate_embeddings(embeddings, training=True)
        
        # Check shape preserved
        assert comm_embeddings.shape == (3, 1, 16)
        assert not torch.isnan(comm_embeddings).any()
    
    def test_get_action(self):
        """Test action selection"""
        config = create_mock_config(n_agents=2)
        agent = MultiAgent_QAC_GAT(config)
        
        observations = [
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32),
            np.array([2.0, 0.3, 20.0, 15.0], dtype=np.float32)
        ]
        
        actions, embeddings = agent.get_action(observations, training=True)
        
        # Check return types
        assert isinstance(actions, list)
        assert len(actions) == 2
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 16)
        
        # Check actions are scalars
        for action in actions:
            assert isinstance(action, (float, np.floating))
            assert 0 <= action <= config.env.max_jobs_per_machine
    
    def test_contrastive_loss_computation(self):
        """Test contrastive loss computation"""
        config = create_mock_config(n_agents=3)
        agent = MultiAgent_QAC_GAT(config)
        
        # Create embeddings at time t and t+1
        embeddings_t = torch.randn(3, 16)
        embeddings_t1 = torch.randn(3, 16)
        
        loss = agent._compute_contrastive_loss(embeddings_t, embeddings_t1)
        
        # Check loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0, "Contrastive loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_contrastive_loss_positive_pairs(self):
        """Test that contrastive loss encourages positive pairs"""
        config = create_mock_config(n_agents=2)
        agent = MultiAgent_QAC_GAT(config)
        
        # Create embeddings where t and t+1 are identical (should have low loss)
        embedding = torch.randn(2, 16)
        embeddings_t = embedding
        embeddings_t1 = embedding.clone()
        
        loss_identical = agent._compute_contrastive_loss(embeddings_t, embeddings_t1)
        
        # Create embeddings where t and t+1 are very different (should have higher loss)
        embeddings_t1_different = -embedding  # Opposite direction
        
        loss_different = agent._compute_contrastive_loss(embeddings_t, embeddings_t1_different)
        
        # Identical embeddings should have lower loss
        assert loss_identical.item() < loss_different.item(), \
            "Identical embeddings should have lower contrastive loss"
    
    def test_update_function(self):
        """Test update function (policy and critic updates)"""
        config = create_mock_config(n_agents=2)
        agent = MultiAgent_QAC_GAT(config)
        
        # Create mock transition
        observations = [
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32),
            np.array([2.0, 0.3, 20.0, 15.0], dtype=np.float32)
        ]
        actions = [15.0, 25.0]
        embeddings = np.random.randn(2, 16).astype(np.float32)
        rewards = [10.5, 12.3]
        next_observations = [
            np.array([1.6, 0.25, 12.0, 16.0], dtype=np.float32),
            np.array([2.1, 0.35, 22.0, 16.0], dtype=np.float32)
        ]
        done = False
        
        # Update
        loss_actor, loss_critic, loss_contrastive = agent.update(
            observations, actions, embeddings, rewards, next_observations, done
        )
        
        # Check return types
        assert isinstance(loss_actor, (float, np.floating))
        assert isinstance(loss_critic, (float, np.floating))
        assert isinstance(loss_contrastive, (float, np.floating))
        assert not np.isnan(loss_actor)
        assert not np.isnan(loss_critic)
        assert not np.isnan(loss_contrastive)
    
    def test_reset_function(self):
        """Test reset function"""
        config = create_mock_config(n_agents=2)
        agent = MultiAgent_QAC_GAT(config)
        
        # Set some state
        agent.prev_embeddings = torch.randn(2, 1, 16)
        
        # Reset
        agent.reset()
        
        assert agent.prev_embeddings is None
    
    def test_multiple_agents_different_actions(self):
        """Test that different agents can select different actions"""
        config = create_mock_config(n_agents=3)
        agent = MultiAgent_QAC_GAT(config)
        
        # Use same observations for all agents (they should still produce different actions due to different networks)
        observations = [
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32),
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32),
            np.array([1.5, 0.2, 10.0, 15.0], dtype=np.float32)
        ]
        
        actions, _ = agent.get_action(observations, training=True)
        
        # Actions should be valid
        assert len(actions) == 3
        for action in actions:
            assert 0 <= action <= config.env.max_jobs_per_machine


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

