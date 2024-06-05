import robosuite as suite
from robosuite.wrappers import GymWrapper
from BottleOpen import BottleOpen
import numpy as np

# Create the environment
env = BottleOpen(
    robots=["Sawyer", "Sawyer"],  # Specifying two robot arms
    use_camera_obs=False,  # Disable camera observations
    has_renderer=True,  # Enable on-screen rendering
    has_offscreen_renderer=False,  # Disable off-screen rendering
    reward_shaping=True,  # Enable reward shaping
)

# Optionally wrap the environment with GymWrapper if you want to use it with Gym
env = GymWrapper(env)

# Reset the environment
obs = env.reset()

# Example usage: Running one episode
done = False
total_reward = 0.0

while not done:
    # Sample a random action
    action = env.action_space.sample()
    action = np.zeros(16)
    # 0-6 are joint angles
    # 7 is the gripper
    # deltas for each actuaator

    # Take a step in the environment
    result = env.step(action)
    obs, obs_dict, reward, done, _, info = result  # Unpack the first four values
    
    # print("oBS: ", [0 if abs(o) < 0.01 else float(o) for o in obs])
    print(obs_dict)

    # Accumulate reward
    total_reward += reward
    
    # Render the environment
    env.render()

print("Total reward:", total_reward)

# Close the environment
env.close()