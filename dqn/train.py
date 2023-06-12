import os

import torch
from ditk import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .agent import Agent
from .base import _GLOBAL_DEVICE


def train_dqn(env, workdir, max_episodes: int = 10000, max_steps: int = 1000, render_per_steps: int = 20,
              eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: float = 0.995,
              replay_buffer_size: int = 10000, batch: int = 64, gamma=0.99, learning_rate=1e-3,
              eval_per_episodes: int = 5, round_per_eval: int = 10, state_trans: str = 'standalone'):
    logging.info(f'Start training for {env!r}, with arguments {locals()!r}')

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    logging.info(f'State dim: {state_dim}, action dim: {action_dim}')

    os.makedirs(workdir, exist_ok=True)
    logging.info(f'Work on directory {workdir!r}')
    tb_writer = SummaryWriter(log_dir=workdir)
    ckpt_dir = os.path.join(workdir, 'ckpts')
    agent = Agent(
        state_dim, action_dim,
        replay_buffer_size, batch, gamma, learning_rate,
        state_trans, _GLOBAL_DEVICE,
    )

    eps = eps_start
    train_step = 0
    for episode in tqdm(range(1, max_episodes + 1)):
        logging.info(f'Episode {episode} start')
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.append(state, action, reward, next_state, done)

            state = next_state
            train_metrics = {
                'eps': eps,
                **agent.train_step(),
            }
            train_step += 1
            logging.info(f'Train step {train_step}, metrics: {train_metrics!r}')
            for key, value in train_metrics.items():
                tb_writer.add_scalar(f'train/{key}', value, train_step)

            # if episode % render_per_steps == 0:
            #     env.render()
            if done:
                break

        agent.save_target()
        eps = max(eps * eps_decay, eps_end)

        if episode % eval_per_episodes == 0:
            rewards = []
            for i in range(1, round_per_eval + 1):
                logging.info(f'Eval at train step {train_step}, round {i}')
                state = env.reset()
                total_reward = 0
                for step in range(max_steps):
                    action = agent.select_action_without_eps(state)
                    next_state, reward, done, _ = env.step(action)

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                rewards.append(total_reward)

            rewards = torch.tensor(rewards).float()
            eval_metrics = {
                'reward_mean': rewards.mean(),
                'reward_max': rewards.max(),
                'reward_min': rewards.min(),
            }
            logging.info(f'Eval metrics at train step {train_step}: {eval_metrics}')
            for key, value in eval_metrics.items():
                tb_writer.add_scalar(f'eval/{key}', value, train_step)

            os.makedirs(ckpt_dir, exist_ok=True)
            last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
            logging.info(f'Saving last ckpt to {last_ckpt!r}')
            torch.save(agent.model.state_dict(), last_ckpt)
