import numpy as np
import time
from bsm import ObstacleAvoidAgent  # Import your ObstacleAvoidAgent
from airsim_env import AirSimEnv  # Import your AirSim environment

# Initialize the gym environment
env = AirSimEnv(vehicle="drone")
env.reset()

# Initialize the obstacle avoidance agent
agent = ObstacleAvoidAgent()

# Define the number of steps to run the simulation
num_steps = 100

for step in range(num_steps):
    # Get the depth image from the environment
    action = {
        'targets': [0, 3, 3, 0],  # Adjust x, y, z velocities as needed
        'active': True
    }
    observation, _, done, _ = env.step(action)  # Assume action [0] returns the current observation
    depth_image = observation['disp']  # Adjust according to how your environment returns the depth image

    # Process the depth image using the obstacle avoidance agent
    agent.image_read(depth_image)
    pitch_offset = 0  # Assuming a pitch offset of 0 for simplicity
    steer, state = agent.steering_behavior(pitch_offset)
    
    # Convert steer to an action the environment understands
    action = {
        'targets': [0, 3, 3, steer],  # Adjust x, y, z velocities as needed
        'active': True
    }
    
    # Perform the action in the environment
    observation, reward, done, _, info = env.step(action)
    
    # Render the environment (optional, can be slow)
    # env.render()

    # Print the current step, state, and steer value for debugging
    print(f"Step: {step}, State: {state}, Steer: {steer}")

    # Check if the episode is done
    if done:
        break

    # Add a small delay to slow down the loop for visualization purposes
    time.sleep(0.1)

# Close the environment
env.close()
