from Qlearning.dqn_keras_rl import create_dqn, set_pommerman_env, create_model, DQN
from Qlearning.env_wrapper import EnvWrapper
from Qlearning.env_with_rewards import EnvWrapperRS
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent

BOARD_SIZE = 11


def env_for_players():
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    agents = [DQN(config["agent"](0, config["game_type"])),
              SimpleAgent(config["agent"](1, config["game_type"])),
              SimpleAgent(config["agent"](2, config["game_type"])),
              SimpleAgent(config["agent"](3, config["game_type"]))]
    env.set_agents(agents)
    env.set_training_agent(agents[0].agent_id)  # training_agent is only dqn agent
    env.set_init_game_state(None)

    return env


def main():
    model = create_model()
    dqn, callbacks = create_dqn(model=model)
    dqn.load_weights('../Qlearning/models/18_03_9-10_new_reward.h5')
    env = EnvWrapperRS(env_for_players(), BOARD_SIZE)  # change env_for_players() to set_pommerman_env to have a simulation
    while True:
        dqn.test(env)


if __name__ == '__main__':
    main()
