import os
import time
import numpy as np

from atari_wrappers import make_atari, wrap_deepmind
from deep_q_network import DeepQNetwork


if __name__ == '__main__':
    LR = 2.5e-4
    HEIGHT = 84
    WIDTH = 84
    NUM_FRAMES = 4
    TEST_EPISODE = 5

    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(par_dir, 'model')
    model_file = os.listdir(MODEL_DIR)[-1]  # 最后保存的模型

    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, scale=False, frame_stack=True)
    num_actions = env.action_space.n

    dqn = DeepQNetwork(input_shape=(WIDTH, HEIGHT, NUM_FRAMES), num_actions=num_actions, name='dqn', learning_rate=LR)
    dqn.load(MODEL_DIR, model_file)

    ep_reward = []
    for _ in range(TEST_EPISODE):
        frame = env.reset()  # LazyFrames
        state = np.array(frame)  # narray (84, 84, 4)
        done = False
        cur_episode_reward = 0
        while not done:  # 如果done则结束episode
            action = dqn.get_action(state / 255.0)
            env.render()
            next_frame, reward, done, _ = env.step(action)
            state = np.array(next_frame)
            cur_episode_reward += reward
            time.sleep(0.005)
        ep_reward.append(cur_episode_reward)
        print('episode: %d - reward: %.2f' % (cur_episode_reward,  cur_episode_reward))

    print('average reward: %.2f' % (np.mean(ep_reward)))
