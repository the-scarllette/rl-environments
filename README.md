# RL-Environments

A collection of simple reinforcement learning environments.
Some are modifications of other environments, others are based on tabletop games,
and some are entirely unique.

## Usage

The environments follow (roughly) the same structure as [OpenAI Gym](https://github.com/openai/gym).
Each environment has a 1 `step` and `reset` method:

```python
def step(action) -> (np.ndarray, float, bool, Any)
```
```python
def reset() -> (np.ndarray, any)
```

Thus an agent-environment loop can be written as:

```python
done = False
state, _ = environment.reset()

while not done:
    action = agent.choose_action(state)
    next_state, rewad, done, _, _ = environment.step(action)
    agent.learn(state, action, reward, next_state, done)
    state = next_state
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
