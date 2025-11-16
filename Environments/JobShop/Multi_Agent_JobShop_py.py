from __future__ import print_function
import numpy as np
import torch
from Src.Utils.Utils import Space

class Multi_Agent_JobShop_py(object):
    """
    Multi-agent version of JobShop environment.
    Each machine is controlled by a separate agent with its own observation and action space.
    """
    def __init__(self,
                 n_machines = 5,
                 n_jobs = 50,
                 debug=False,
                 max_steps=100,
                 mappingType ='DNC'
                 ):

        self.n_actions = n_machines  # Number of agents (machines)
        self.max_steps = max_steps
       
        self.num_machines = n_machines  # Number of machines/agents
        self.max_jobs_per_machine = n_jobs  # Maximum capacity per machine
        self.max_energy_usage = 100  # Max energy that can be used by a machine
        
        # Initialize state
        self.machines = {'energy_usage': np.random.uniform(1.0, 1.2, self.num_machines),
                         'wear_level': np.zeros(self.num_machines)}  # machines are heterogeneous
        # Job queue length per machine (for multi-agent observations)
        self.job_queue_lengths = np.zeros(self.num_machines)

        
        if mappingType == 'knn_mapping' or mappingType == 'learned_mapping' or mappingType == 'no_mapping':
            raise Exception("Sorry, this mapping type is not supported") 
        else:
            self.actionLiteral = True
        
        self.debug = debug
      
        # Per-agent action space (scalar): a_i ∈ [0, n_jobs]
        self.action_space = Space(low=np.array([0.0], dtype=np.float32), 
                                 high=np.array([float(n_jobs)], dtype=np.float32), 
                                 dtype=np.float32)
        # Per-agent observation space: [e_i, w_i, queue_length_i, global_avg_load]
        self.observation_space = Space(low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
                                     high=np.array([self.max_energy_usage, 5.0, float(n_jobs), float(n_jobs)], dtype=np.float32),
                                     dtype=np.float32)

        self.action_space_matrix = []
        
        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        print('not implemented')
        return None

    def step(self, actions, training=True):
        """
        Multi-agent step function.
        
        Args:
            actions: List/array of n_machines scalar actions [a_1, a_2, ..., a_n]
        
        Returns:
            observations: List of per-agent observations
            rewards: List of per-agent rewards (mixed local + global)
            done: Boolean
            info: Dict with additional info (global metrics)
        """
        # Convert actions to numpy array if needed
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        elif isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # Clip actions to valid range
        actions = np.clip(actions, 0.0, float(self.max_jobs_per_machine))
        
        job_counts = []
        local_rewards = []
        
        # Update each machine based on its agent's action
        for i, jobs in enumerate(actions):
            # Update wear level
            wear_increase = np.random.uniform(0.05, 0.4) if jobs > 0.75 * self.max_jobs_per_machine else 0
            self.machines['wear_level'][i] += wear_increase
            
            if jobs < 0.5 * self.max_jobs_per_machine:
                wear_decrease = np.random.uniform(0.05, 0.4)
                self.machines['wear_level'][i] = max(self.machines['wear_level'][i] - wear_decrease, 0)
            
            # Update energy usage
            self.machines['energy_usage'][i] = self.calculate_energy_usage(i, jobs)
            
            # Update job queue length (simplified: current action becomes new queue)
            self.job_queue_lengths[i] = jobs
            
            # Calculate local reward component
            energy_cost = self.machines['energy_usage'][i]
            wear_cost = self.machines['wear_level'][i] * 0.5  # Scale wear cost
            throughput_reward = jobs * 3
            r_local = -energy_cost - wear_cost + throughput_reward
            
            local_rewards.append(r_local)
            job_counts.append(jobs)
        
        # Calculate global reward components
        job_counts = np.array(job_counts)  # Convert to numpy array
        load_imbalance = np.std(job_counts)  # Penalty for uneven load
        avg_load = np.mean(job_counts)
        idle_penalty = np.sum(np.maximum(0, 0.1 * self.max_jobs_per_machine - job_counts))  # Penalty for idle machines
        overload_penalty = np.sum(np.maximum(0, job_counts - 0.9 * self.max_jobs_per_machine))  # Penalty for overload
        
        r_global = -load_imbalance - 0.1 * idle_penalty - 0.1 * overload_penalty
        
        # Mixed reward: α * r_local + (1-α) * r_global (α ≈ 0.7)
        alpha = 0.7
        rewards = [alpha * r_loc + (1 - alpha) * r_global for r_loc in local_rewards]
        
        self.current_step += 1
        done = (self.current_step >= self.max_steps - 1)
        
        # Get per-agent observations
        observations = self._next_observation()
        
        info = {
            'global_load_imbalance': load_imbalance,
            'avg_load': avg_load,
            'total_energy': np.sum(self.machines['energy_usage']),
            'total_wear': np.sum(self.machines['wear_level'])
        }
        
        return observations, rewards, done, info

    def calculate_energy_usage(self, machine_id, jobs):
        base_energy = self.machines['energy_usage'][machine_id]  # Base energy consumption rate
        wear_factor = self.machines['wear_level'][machine_id]
        energy_usage = base_energy * (1 + wear_factor) * jobs
        return min(energy_usage, self.max_energy_usage)  # Cap the energy usage

    def reset(self, training=False):
        self.machines = {'energy_usage': np.random.uniform(1.0, 2, self.num_machines),
                         'wear_level': np.zeros(self.num_machines)}
        self.current_step = 0
        self.job_queue_lengths = np.zeros(self.num_machines)
        return self._next_observation()

    def _next_observation(self):
        """
        Generate per-agent observations for multi-agent mode.
        Each agent observes: [e_i, w_i, queue_length_i, global_avg_load]
        """
        global_avg_load = np.mean(self.job_queue_lengths)
        observations = []
        
        for i in range(self.num_machines):
            obs_i = np.array([
                self.machines['energy_usage'][i],
                self.machines['wear_level'][i],
                self.job_queue_lengths[i],
                global_avg_load
            ], dtype=np.float32)
            observations.append(obs_i)
        
        return observations

    def render(self, mode='human'):
        print(f"Machine states: {self.machines}")
        print(f"Job queue lengths: {self.job_queue_lengths}")

 
if __name__=="__main__":
    # Test Multi-Agent JobShop
    env = Multi_Agent_JobShop_py(n_machines=5, n_jobs=50)
    observations = env.reset()
    done = False
    total_rewards = [0.0] * env.num_machines
    
    while not done:
        # Random actions for each agent
        actions = np.random.uniform(0, 50, env.num_machines)
        observations, rewards, done, info = env.step(actions)
        for i, r in enumerate(rewards):
            total_rewards[i] += r
        env.render()
        print(f"Rewards: {rewards}")
        print(f"Info: {info}")
    
    print(f"Total rewards per agent: {total_rewards}")

