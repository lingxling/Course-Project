import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from atari_wrappers import make_atari, wrap_deepmind
from utils import epsilon_greedy
from deep_q_network import DeepQNetwork
from memory_buffer import MemoryBuffer

if __name__ == '__main__':
    MIN_BUFFER = 10000
    BUFFER_SIZE = 10001
    MAX_EPISODE = 1000
    MINI_BATCH = 32  # [16, 32, 64]
    GAMMA = 0.99
    C = 1000
    LR = 2.5e-4
    HEIGHT = 84
    WIDTH = 84
    NUM_FRAMES = 4

    # 保存数据的文件夹
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(par_dir, 'model')
    IMG_DIR = os.path.join(par_dir, 'img')
    RES_DIR = os.path.join(par_dir, 'res')
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    if not os.path.exists(RES_DIR):
        os.mkdir(RES_DIR)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # 设置gym有关参数
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, scale=False, frame_stack=True)
    num_actions = env.action_space.n

    dqn = DeepQNetwork(input_shape=(WIDTH, HEIGHT, NUM_FRAMES), num_actions=num_actions, name='dqn', learning_rate=LR)
    target_dqn = DeepQNetwork(input_shape=(WIDTH, HEIGHT, NUM_FRAMES), num_actions=num_actions, name='target_dqn', learning_rate=LR)
    buf = MemoryBuffer(memory_size=BUFFER_SIZE)

    total_episode_rewards = []
    step = 0
    for episode in range(MAX_EPISODE + 1):
        frame = env.reset()  # LazyFrames
        state = np.array(frame)  # narray (84, 84, 4)
        done = False
        cur_episode_reward = 0
        while not done:  # 如果done则结束episode
            if step % C == 0:
                target_dqn.copy_from(dqn)  # 复制参数
            if epsilon_greedy(step):
                action = env.action_space.sample()
            else:
                action = dqn.get_action(state / 255.0)
            # env.render()
            next_frame, reward, done, _ = env.step(action)
            next_state = np.array(next_frame)
            buf.push(state, action, reward, next_state, done)
            state = next_state
            cur_episode_reward += reward

            if buf.size() > MIN_BUFFER:
                states, actions, rewards, next_states, dones = buf.sample(MINI_BATCH)
                next_state_action_values = np.max(target_dqn.predict(next_states / 255.0), axis=1)
                y_true = dqn.predict(states / 255.0)  # Y.shape: (MINI_BATCH, num_actions), i.e., (32, 6)
                y_true[range(MINI_BATCH), actions] = rewards + GAMMA * next_state_action_values * np.invert(dones)
                dqn.train(states / 255.0, y_true)
            step += 1
        total_episode_rewards.append(cur_episode_reward)
        if episode % 100 == 0:
            dqn.save(MODEL_DIR, 'dqn-{}'.format(episode))
        if np.mean(total_episode_rewards[-30:]) > 19:
            dqn.save(MODEL_DIR, 'dqn-{}'.format(episode))
            break
    np.save(os.path.join(RES_DIR, 'episode_rewards.npy'), np.array(total_episode_rewards))

    # 画episode_reward
    plt.figure()
    plt.title('EPISODE - REWARD')
    plt.plot(range(len(total_episode_rewards)), total_episode_rewards, linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(os.path.join(IMG_DIR, 'episode_reward.png'))

    # 画mean_100_episode_reward
    mean_100_reward = []
    for i in range(1, len(total_episode_rewards)):
        mean_100_reward.append(np.mean(total_episode_rewards[max(0, i - 100):i]))
    plt.figure()
    plt.title('MEAN 100 EPISODE - REWARD')
    plt.plot(range(1, len(total_episode_rewards)), mean_100_reward, linewidth=2, color='r')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(os.path.join(IMG_DIR, 'mean_100_episode_reward.png'))

