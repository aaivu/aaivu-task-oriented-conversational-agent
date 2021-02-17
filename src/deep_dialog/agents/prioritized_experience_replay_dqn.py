import math
import json
import pickle
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from deep_dialog import dialog_config

from .agent import Agent
from .prioritized_replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = 10000  # replay buffer size
BATCH_SIZE = 16  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_NN_EVERY = 1  # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 20  # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 1000  # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = 85 * BATCH_SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class  PrioritizedDQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, seed=1, phase='training'):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        self.running_phase = phase

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)
        self.model_path = params['model_path']

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + \
                               7 * self.slot_cardinality + 3 + self.max_turn

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(
            self.state_dimension, self.num_actions, seed).to(self.device)

        if self.running_phase != 'inference':
            self.epsilon = params['epsilon']
            self.agent_run_mode = params['agent_run_mode']
            self.agent_act_level = params['agent_act_level']
            self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

            self.t_step = 0

            self.experience_replay_pool_size = params.get(
                'experience_replay_pool_size', 1000)
            self.hidden_size = params.get('dqn_hidden_size', 60)
            self.gamma = params.get('gamma', 0.9)
            self.predict_mode = params.get('predict_mode', False)
            self.warm_start = params.get('warm_start', 0)

            # Replay memory
            self.memory = PrioritizedReplayBuffer(
                self.num_actions, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, seed, compute_weights)
            # Initialize time step (for updating every UPDATE_NN_EVERY steps)
            self.t_step_nn = 0
            # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
            self.t_step_mem_par = 0
            # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
            self.t_step_mem = 0

            self.qnetwork_target = QNetwork(
                self.state_dimension, self.num_actions, seed).to(self.device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

            self.cur_bellman_err = 0

    def load_model(self):
        self.qnetwork_local.load_state_dict(torch.load(self.model_path))
        self.qnetwork_local.eval()

    def save_model(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime',
                            'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + \
            kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]
                             ] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + \
            np.sum(kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(
                    kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep,
                                               agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation



    def run_policy(self, state):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                state = torch.from_numpy(
                    state).float().unsqueeze(0).to(self.device)
                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local(state)
                    self.qnetwork_local.train()
                return np.argmax(action_values.cpu().data.numpy())


    def rule_policy(self):
        """ Rule Policy """

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {
                'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks",
                                 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)


    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None


    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """

        state = self.prepare_state_representation(s_t)
        action = self.action
        reward = reward
        next_state = self.prepare_state_representation(s_tplus1)
        done = episode_over

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.memory.add(state, action, reward, next_state, done)
        else:  # Prediction Mode
            self.memory.add(state, action, reward, next_state, done)



    def train(self, batch_size=16, num_batches=100, gamma = 0.99):
        """ Train DQN with experience replay """
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            self.memory.update_memory_sampling()
            self.memory.update_parameters()
            for iter in range(int(EXPERIENCES_PER_SAMPLING/batch_size)):
                experiences = self.memory.sample()
                self.learn(experiences, gamma)


    def learn(self, sampling, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = sampling

        ## TODO: compute and minimize the loss
        q_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        expected_values = rewards + gamma * q_target * (1 - dones)
        output = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(output, expected_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update priorities ------------------- #
        delta = abs(expected_values - output.detach()).numpy()
        self.memory.update_priorities(delta, indices)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def reinitialize_memory(self):
        self.memory = PrioritizedReplayBuffer(
            self.num_actions, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, self.seed, self.compute_weights)

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        return model
