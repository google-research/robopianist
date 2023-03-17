# RoboPianist Suite

![Task Suite](robopianist.png)

## Quickstart

```python
import numpy as np
from robopianist import suite

# Print out all available tasks.
print(suite.ALL)

# Print out robopianist-etude-12 task subset.
print(suite.ETUDE_12)

# Load an environment from the debug subset.
env = suite.load("RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
action_spec = env.action_spec()

# Step through an episode and print out the reward, discount and observation.
timestep = env.reset()
while not timestep.last():
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape
    ).astype(action_spec.dtype)
    timestep = env.step(action)
    print(timestep.reward, timestep.discount, timestep.observation)
```
