from Qlearning.dqn_keras_rl import create_dqn, set_pommerman_env, create_model, DQN
from Qlearning.env_wrapper import EnvWrapper
from Qlearning.env_with_rewards import EnvWrapperRS
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent

from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QIcon
import pickle
from datetime import datetime
import os
import pommerman
from pommerman import agents



class Menu(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.background = QLabel(self)
        self.start_button = QPushButton('', self)
        self.select_game_screen = SelectGame(self)
        self.load_game_button = QPushButton('', self)

        self.init_gui()

    def init_gui(self):
        self.select_game_screen.hide()
        self.setGeometry(200, 200, 512, 480)
        self.setWindowTitle('BomberMan')
        self.background.setPixmap(QPixmap('../pommerman/resources/Menu_screen2.png'))

        start_button_pix = QPixmap('../pommerman/resources/Start_Button.png')
        load_button_pix = QPixmap('../pommerman/resources/load_game.png')

        create_button(self.start_button, start_button_pix, 212, 347)
        create_button(self.load_game_button, load_button_pix, 212, 387)
        self.start_button.clicked.connect(self.start)
        self.load_game_button.clicked.connect(self.load_game)

    def start(self):
        self.hide()
        self.select_game_screen.show()

    def load_game(self):
        self.hide()
        with open('./replay/000.pickle', 'rb') as f:
            replay_game = pickle.load(f)
        num_players = replay_game.pop()
        agents_list = [agents.RandomAgent() for i in range(num_players)]

        env = pommerman.make('PommeFFACompetition-v0', agents_list,
                             game_state_file='./replay/000.json')

        # Run the episodes just like OpenAI Gym
        env.reset()
        for actions in replay_game:
            env.render()
            env.step(actions)
        env.close()
        self.show()


class SelectGame(QWidget):
    def __init__(self, menu, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.background = QLabel(self)
        self.player1_button = QPushButton('', self)
        self.player2_button = QPushButton('', self)
        self.player3_button = QPushButton('', self)
        self.simulation_button = QPushButton('', self)
        self.game_over_screen = GameOver(self, menu)
        self.game_replay = []

        self.init_gui()

    def init_gui(self):
        self.game_over_screen.hide()
        self.setGeometry(200, 200, 512, 480)
        self.setWindowTitle('BomberMan')
        self.background.setPixmap(QPixmap('../pommerman/resources/Menu_screen2.png'))

        player1pix = QPixmap('../pommerman/resources/player1.png')
        player2pix = QPixmap('../pommerman/resources/player2.png')
        player3pix = QPixmap('../pommerman/resources/player3.png')
        simulationpix = QPixmap('../pommerman/resources/simulation.png')

        create_button(self.player1_button, player1pix, 40, 331)
        create_button(self.player2_button, player2pix, 40, 371)
        create_button(self.player3_button, player3pix, 280, 331)
        create_button(self.simulation_button, simulationpix, 280, 371)

        self.player1_button.clicked.connect(self.player_vs_1)
        self.player2_button.clicked.connect(self.player_vs_2)
        self.player3_button.clicked.connect(self.player_vs_3)
        self.simulation_button.clicked.connect(self.simulation)

    def player_vs_1(self):
        self.hide()
        self.start_gameplay(2, False)

    def player_vs_2(self):
        self.hide()
        self.start_gameplay(3, False)

    def player_vs_3(self):
        self.hide()
        # self.start_gameplay(4, False)
        self.start_game_with_agent(4)

    def simulation(self):
        self.hide()
        # self.start_gameplay(4, True)
        self.simulation_with_agent()

    def start_game_with_agent(self, num_players=4):

        model = create_model()
        dqn, callbacks = create_dqn(model=model)
        dqn.load_weights('../Qlearning/models/18_03_9-10_new_reward.h5')
        env = EnvWrapperRS(self.env_for_players(),
                           11)  # change env_for_players() to set_pommerman_env to have a simulation

        dqn.test(env)
        env.close()
        self.game_replay.append(num_players)
        self.game_over_screen.show()

    def simulation_with_agent(self):
        model = create_model()
        dqn, callbacks = create_dqn(model=model)
        dqn.load_weights('../Qlearning/models/18_03_9-10_new_reward.h5')
        env = EnvWrapperRS(self.env_for_simulation(),
                           11)  # change env_for_players() to set_pommerman_env to have a simulation
        while True:
            dqn.test(env)
        env.close()

    def start_gameplay(self, num_players=4, simulation=True):
        # Create a set of agents (exactly four)
        agent_list = [agents.SimpleAgent() for i in range(num_players)]

        if not simulation:
            agent_list[0] = agents.PlayerAgent()

        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetition-v0', agent_list)

        # Run the episodes just like OpenAI Gym
        for i_episode in range(1):
            state = env.reset()
            env.save_json('./replay')
            done = False
            while not done:
                env.render()
                actions = env.act(state)
                self.game_replay.append(actions)
                state, reward, done, info = env.step(actions)
        env.close()
        self.game_replay.append(num_players)
        self.game_over_screen.show()

    def env_for_players(self):
        config = ffa_v0_fast_env(30)
        env = Pomme(**config["env_kwargs"])
        agents = [DQN(config["agent"](0, config["game_type"])),
                  PlayerAgent(config["agent"](1, config["game_type"])),
                  RandomAgent(config["agent"](2, config["game_type"])),
                  RandomAgent(config["agent"](3, config["game_type"]))]
        env.set_agents(agents)
        env.set_training_agent(agents[0].agent_id)  # training_agent is only dqn agent
        env.set_init_game_state(None)

        return env

    def env_for_simulation(self):
        config = ffa_v0_fast_env(30)
        env = Pomme(**config["env_kwargs"])
        agents = [DQN(config["agent"](0, config["game_type"])),
                  SimpleAgent(config["agent"](1, config["game_type"])),
                  SimpleAgent(config["agent"](2, config["game_type"])),
                  SimpleAgent(config["agent"](3, config["game_type"]))]
        env.set_agents(agents)
        env.set_training_agent(agents[0].agent_id)  # training_agent is only dqn agent
        env.set_init_game_state(None)

        return env


def create_button(button, button_pix, x, y):
    button.setIcon(QIcon(button_pix))
    button.setIconSize(button_pix.size())
    button.setMaximumSize(button_pix.size())
    button.setMinimumSize(button_pix.size())
    button.move(x, y)


class GameOver(QWidget):
    def __init__(self, game_screen, menu, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.menu = menu
        self.game_screen = game_screen
        self.mainframe = QLabel('', self)
        self.continue_button = QPushButton('', self)
        self.save_button = QPushButton('', self)
        self.init_gui()

    def init_gui(self):
        game_over_png = QPixmap('../pommerman/resources/game_over.png')
        self.mainframe.setPixmap(game_over_png)
        self.setGeometry(200, 200, game_over_png.size().width(), game_over_png.size().height())
        self.setWindowTitle('Game Over')
        continue_png = QPixmap('../pommerman/resources/continue.png')
        save_png = QPixmap('../pommerman/resources/save.png')

        create_button(self.continue_button, continue_png, 172, 355)
        create_button(self.save_button, save_png, 173, 393)

        self.continue_button.clicked.connect(self.continue_action)
        self.save_button.clicked.connect(self.save)

    def continue_action(self):
        self.hide()
        self.menu.show()

    def save(self):
        path = './replay'
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError:
                print(f"Creation of directory {path} failed")

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = path+'/'+time+'.pickle'
        with open('./replay/000.pickle', 'wb') as f:
            pickle.dump(self.game_screen.game_replay, f,  protocol=pickle.HIGHEST_PROTOCOL)
        self.hide()
        self.menu.show()


if __name__ == '__main__':
    app = QApplication([])
    menu = Menu()
    menu.show()
    app.exec_()
