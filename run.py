from SteeringBehaviors import Wander, Seek
import SimulationEnvironment as sim
from Networks import Action_Conditioned_FF

import pickle
import numpy as np
import torch
import numpy.linalg as la


def get_network_param(sim_env, action, scaler):
    sensor_readings = sim_env.raycasting()
    network_param = np.append(sensor_readings, [action, 0]) #unutilized 0 added to match shape of scaler
    network_param = scaler.transform(network_param.reshape(1,-1))
    network_param = network_param.flatten()[:-1]
    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param

def goal_seeking(goals_to_reach):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 20
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)

    #load model
    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load('saved/saved_model.pkl'))
    model.eval()

    #load normalization parameters
    scaler = pickle.load( open("saved/scaler.pkl", "rb"))

    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    robot_turned_around = False
    actions_checked = []
    goals_reached = 0
    last_position = None
    stuck_counter = 0
    collision_threshold = 0.25  # Start with conservative threshold
    consecutive_no_actions = 0  # Track consecutive iterations with no safe actions
    while goals_reached < goals_to_reach:

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        if la.norm(seek_vector) < 50:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            print("goal reached +1")
            goals_reached += 1
            continue

        action_space = np.arange(-5,6)
        actions_available = []
        action_predictions = {}  # Store predictions for all actions
        
        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler)
            prediction = model(network_param)
            # Apply sigmoid to convert logits to probabilities (0-1 range)
            # Lower probability = safer action (less likely to collide)
            prediction_value = torch.sigmoid(prediction).item()
            # Handle NaN predictions
            if np.isnan(prediction_value):
                prediction_value = 1.0  # Treat NaN as unsafe
            action_predictions[action] = prediction_value
            if prediction_value < collision_threshold:
                actions_available.append(action)

        # Get desired action from steering behavior
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        
        if len(actions_available) == 0:
            # If no actions are available, gradually increase threshold to allow more actions
            consecutive_no_actions += 1
            if consecutive_no_actions > 5:
                # Gradually increase threshold when stuck
                collision_threshold = min(0.75, collision_threshold + 0.1)
                consecutive_no_actions = 0
            else:
                # Pick actions with lowest collision predictions
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                if len(valid_predictions) == 0:
                    valid_predictions = action_predictions
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                # Take the top 3 safest actions
                safest_actions = [a[0] for a in sorted_actions[:3]]
                actions_available = safest_actions
            
            # Increment stuck counter when no safe actions are available
            stuck_counter += 1
            # Also limit how often we turn around to prevent infinite loops
            if stuck_counter > 20:  # If we've been stuck for a while, force a turn
                sim_env.turn_robot_around()
                stuck_counter = 0
                last_position = None
                continue
        else:
            # Reset counters when we have safe actions
            consecutive_no_actions = 0
            # Gradually lower threshold back to conservative when we have safe actions
            if collision_threshold > 0.25:
                collision_threshold = max(0.25, collision_threshold - 0.05)
        
        # Find the action closest to the desired steering direction
        if len(actions_available) == 0:
            # Should not happen, but handle it
            closest_action = 0
        else:
            min_diff, closest_action = 9999, actions_available[0]
            for a in actions_available:
                diff = abs(desired_action - a)
                if diff < min_diff:
                    min_diff = diff
                    closest_action = a
        
        # If the closest action is 0 (straight ahead) and we have other options,
        # prefer a slightly turning action to ensure forward movement
        if closest_action == 0 and len(actions_available) > 1:
            # Prefer actions that are close to the desired action but not exactly 0
            # This helps avoid getting stuck in place
            for a in actions_available:
                if a != 0 and abs(action - a) <= min + 1:
                    closest_action = a
                    break

        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        
        # Check if robot is stuck (not moving)
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < 5.0:  # Robot hasn't moved much
                stuck_counter += 1
                if stuck_counter > 10:  # Robot has been stuck for a while
                    # Force a turn to get unstuck
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    last_position = None
                    continue
            else:
                stuck_counter = 0  # Reset counter if robot is moving
        last_position = current_position
        
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                steering_behavior.reset_action()
                stuck_counter = 0  # Reset stuck counter on collision
                last_position = None
                break


if __name__ == '__main__':
    goals_to_reach = 2
    goal_seeking(goals_to_reach)
