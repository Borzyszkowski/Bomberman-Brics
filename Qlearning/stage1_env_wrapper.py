from env_wrapper import EnvWrapper
import time

class StageOneEnvWrapper(EnvWrapper):
    def __init__(self, gym, board_size):
        super(StageOneEnvWrapper, self).__init__(gym, board_size)
        self.previous_board_features = None
        self.previous_obs = None
        self.steps_counter = 0
        self.steps_with_no_bombs = 0
        self.steps_with_no_wall_destroy = 0
        self.steps_with_no_move = 0

        self.steps_limit = 1000
        self.destroyable_wall_on_board = 2
        self.agent_on_board = 10
        self.enemies_on_board = 11

        self.max_steps_with_no_move = 4
        self.no_move_reward = -0.05
        self.destroy_wall_reward = 0.1
        self.die_reward = -1
        self.win_reward = 1
        self.max_steps_with_no_bombs = 6
        self.no_bombs_reward = -0.1
        self.max_steps_with_no_wall_destroy = 12
        self.no_wall_destroy_reward = -0.1

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)

        action = all_actions[self.gym.training_agent]
        agent_state = self.custom_featurize(state[self.gym.training_agent])
        # agent_reward = reward[self.gym.training_agent]
        
        agent_reward = 0
        agent_reward, terminal = self.rewarding(obs, agent_reward)

        self.steps_counter += 1

        return agent_state, agent_reward, terminal, info

    def rewarding(self, obs, reward):

        board_features = obs[self.gym.training_agent]['board'].copy()
        # get board coordinates of our agent
        agent_y, agent_x = obs[self.gym.training_agent]['position']
        for i in range(len(board_features)):
            for j in range(len(board_features[i])):
                # set enemies on board to 11
                if 10 <= board_features[i][j] <= 13:
                    board_features[i][j] = self.enemies_on_board
                # set our agent to 10
                if i == agent_y and j == agent_x:
                    board_features[i][j] = self.agent_on_board

        # If every wall was destroyed
        if self.count_numbers_on_board(board_features, self.destroyable_wall_on_board) == 0:
            terminal = True
            reward += self.win_reward
        else:
            terminal = False

        # If there was max step
        if self.steps_counter >= self.steps_limit:
            terminal = True
            reward += self.die_reward

        # If agent death
        if self.agent_on_board not in obs[self.gym.training_agent]['alive']:
            terminal = True
            reward += self.die_reward

        # If it's first step
        if self.previous_board_features is None:
            self.previous_board_features = board_features
            self.previous_obs = obs.copy()
            return reward, terminal

        previous_destroyable_walls_number = self.count_numbers_on_board(self.previous_board_features, self.destroyable_wall_on_board)
        current_destroyable_walls_number = self.count_numbers_on_board(board_features, self.destroyable_wall_on_board)

        # if player destroy wall
        if previous_destroyable_walls_number > current_destroyable_walls_number:
            reward += (self.destroy_wall_reward * (previous_destroyable_walls_number - current_destroyable_walls_number))
            self.steps_with_no_wall_destroy = 0
        else:
            self.steps_with_no_wall_destroy += 1
            if self.steps_with_no_wall_destroy > self.max_steps_with_no_wall_destroy:
                reward += self.no_wall_destroy_reward
                self.steps_with_no_wall_destroy = 0

        # If agent doesnt move
        if obs[self.gym.training_agent]['position'] == self.previous_obs[self.gym.training_agent]['position']:
            self.steps_with_no_move += 1
            if self.steps_with_no_move > self.max_steps_with_no_move:
                reward += self.no_move_reward
                self.steps_with_no_move = 0
        else:
            self.steps_with_no_move = 0

        print(reward)
        time.sleep(0.5)

        self.previous_board_features = board_features
        self.previous_obs = obs.copy()

        return reward, terminal

    def count_numbers_on_board(self, board_features, number_to_count):
        counter = 0
        for row in board_features:
            for element in row:
                if element == number_to_count:
                    counter += 1

        return counter

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        obs = self.gym.reset()
        agent_obs = self.featurize(obs[self.gym.training_agent])
        self.previous_board_features = None
        self.previous_obs = None
        self.steps_counter = 0
        self.steps_with_no_bombs = 0
        self.steps_with_no_wall_destroy = 0
        self.steps_with_no_move = 0
        return agent_obs
