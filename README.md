# dqn_demo

DQN demo for cliff walking

```python
import gym
from ditk import logging

from dqn import train_dqn

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    env = gym.make("CliffWalking-v0")

    train_dqn(
        env=env,
        workdir='runs/first_demo',

        max_episodes=10000,
        max_steps=1000,
        render_per_steps=20,
        eval_per_episodes=5,
        round_per_eval=10,

        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.995,
        replay_buffer_size=10000,

        batch=64,
        gamma=0.99,
        learning_rate=1e-3,
        state_trans='standalone'
    )

```