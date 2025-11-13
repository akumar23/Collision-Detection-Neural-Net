import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Wander
from src.robot_navigation import simulation as sim

import numpy as np


def collect_training_data(total_actions):
    # set-up environment
    sim_env = sim.SimulationEnvironment()

    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7

    network_params = []

    for action_i in range(total_actions):
        progress = 100 * float(action_i) / total_actions
        #print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        # steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()

                if action_timestep < action_repeat * .3:  # in case prior action caused collision
                    network_params[-1][-1] = collision  # share collision result with prior action
                break
        print(sensor_readings)
        print(action)
        print(collision)
        sensor_readings = np.append(sensor_readings, action)
        sensor_readings = np.append(sensor_readings, collision)
        print(sensor_readings)
        print(action)
        print(collision)

        network_params.append(sensor_readings)

    # Save to data directory
    data_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    np.savetxt(data_path, network_params, delimiter=",")


if __name__ == '__main__':
    # Collect more training data for better model accuracy
    # Current: 50,000 samples, increasing to 100,000 for better coverage
    total_actions = 100000
    print(f"Collecting {total_actions} training samples...")
    print("This may take a while. Progress will be shown.")
    collect_training_data(total_actions)
    print(f"\nTraining data collection complete! Collected {total_actions} samples.")
