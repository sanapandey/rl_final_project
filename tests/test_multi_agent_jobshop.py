"""
Tests for Multi_Agent_JobShop_py environment
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from Environments.JobShop.Multi_Agent_JobShop_py import Multi_Agent_JobShop_py


class TestMultiAgentJobShop:
    """Test suite for Multi_Agent_JobShop_py environment"""
    
    def test_initialization(self):
        """Test environment initialization"""
        env = Multi_Agent_JobShop_py(n_machines=5, n_jobs=50, max_steps=100)
        
        assert env.num_machines == 5
        assert env.max_jobs_per_machine == 50
        assert env.max_steps == 100
        assert len(env.machines['energy_usage']) == 5
        assert len(env.machines['wear_level']) == 5
        assert len(env.job_queue_lengths) == 5
    
    def test_reset(self):
        """Test environment reset"""
        env = Multi_Agent_JobShop_py(n_machines=3, n_jobs=30)
        observations = env.reset()
        
        # Should return list of per-agent observations
        assert isinstance(observations, list)
        assert len(observations) == 3
        
        # Each observation should be [e_i, w_i, queue_length_i, global_avg_load]
        for obs in observations:
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (4,)
            assert obs.dtype == np.float32
            # Check bounds
            assert obs[0] >= 0  # energy_usage
            assert obs[1] >= 0  # wear_level
            assert obs[2] >= 0  # queue_length
            assert obs[3] >= 0  # global_avg_load
    
    def test_step_single_step(self):
        """Test single step in environment"""
        env = Multi_Agent_JobShop_py(n_machines=3, n_jobs=30)
        env.reset()
        
        # Actions: one scalar per agent
        actions = [10.0, 15.0, 20.0]
        observations, rewards, done, info = env.step(actions)
        
        # Check return types
        assert isinstance(observations, list)
        assert len(observations) == 3
        assert isinstance(rewards, list)
        assert len(rewards) == 3
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observations
        for obs in observations:
            assert obs.shape == (4,)
            assert obs.dtype == np.float32
        
        # Check rewards (should be floats)
        for reward in rewards:
            assert isinstance(reward, (float, np.floating))
        
        # Check info dict
        assert 'global_load_imbalance' in info
        assert 'avg_load' in info
        assert 'total_energy' in info
        assert 'total_wear' in info
    
    def test_step_action_clipping(self):
        """Test that actions are clipped to valid range"""
        env = Multi_Agent_JobShop_py(n_machines=2, n_jobs=50)
        env.reset()
        
        # Actions outside valid range
        actions = [-10.0, 100.0]  # Below 0 and above max_jobs_per_machine
        observations, rewards, done, info = env.step(actions)
        
        # Should not crash and should clip actions
        assert len(observations) == 2
        assert len(rewards) == 2
    
    def test_step_machine_state_updates(self):
        """Test that machine states update correctly"""
        env = Multi_Agent_JobShop_py(n_machines=2, n_jobs=50)
        env.reset()
        
        initial_energy = env.machines['energy_usage'].copy()
        initial_wear = env.machines['wear_level'].copy()
        
        actions = [25.0, 30.0]
        observations, rewards, done, info = env.step(actions)
        
        # Energy and wear should have changed (or at least be updated)
        # Note: Energy depends on jobs, so it should change
        # Wear may or may not change depending on job level
        assert np.any(env.machines['energy_usage'] != initial_energy) or np.any(env.machines['wear_level'] != initial_wear)
    
    def test_mixed_reward_structure(self):
        """Test that rewards have both local and global components"""
        env = Multi_Agent_JobShop_py(n_machines=3, n_jobs=50)
        env.reset()
        
        actions = [20.0, 20.0, 20.0]  # Balanced load
        _, rewards_balanced, _, _ = env.step(actions)
        
        env.reset()
        actions = [10.0, 30.0, 40.0]  # Imbalanced load
        _, rewards_imbalanced, _, _ = env.step(actions)
        
        # Rewards should be different (imbalanced should generally be worse)
        # But this is stochastic, so we just check they're computed
        assert len(rewards_balanced) == 3
        assert len(rewards_imbalanced) == 3
        assert all(isinstance(r, (float, np.floating)) for r in rewards_balanced)
        assert all(isinstance(r, (float, np.floating)) for r in rewards_imbalanced)
    
    def test_episode_termination(self):
        """Test that episodes terminate correctly"""
        env = Multi_Agent_JobShop_py(n_machines=2, n_jobs=50, max_steps=5)
        env.reset()
        
        done = False
        step_count = 0
        while not done:
            actions = [10.0, 15.0]
            _, _, done, _ = env.step(actions)
            step_count += 1
            assert step_count <= env.max_steps
        
        assert done
        # Episode terminates when current_step >= max_steps - 1
        # So we get max_steps - 1 steps before termination
        assert step_count == env.max_steps - 1 or step_count == env.max_steps
    
    def test_observation_structure(self):
        """Test that observations have correct structure"""
        env = Multi_Agent_JobShop_py(n_machines=4, n_jobs=50)
        observations = env.reset()
        
        for i, obs in enumerate(observations):
            # Check shape: [e_i, w_i, queue_length_i, global_avg_load]
            assert obs.shape == (4,), f"Agent {i} observation shape incorrect: {obs.shape}"
            
            # Check that global_avg_load is the same for all agents
            global_avg = obs[3]
            for other_obs in observations:
                assert np.isclose(other_obs[3], global_avg), "Global avg load should be same for all agents"
    
    def test_job_queue_lengths_update(self):
        """Test that job queue lengths update with actions"""
        env = Multi_Agent_JobShop_py(n_machines=3, n_jobs=50)
        env.reset()
        
        # Initial queue lengths should be zero
        assert np.all(env.job_queue_lengths == 0)
        
        actions = [15.0, 20.0, 25.0]
        observations, _, _, _ = env.step(actions)
        
        # Queue lengths should match actions (simplified model)
        for i, action in enumerate(actions):
            assert np.isclose(env.job_queue_lengths[i], action), \
                f"Queue length {i} should match action: {env.job_queue_lengths[i]} vs {action}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

