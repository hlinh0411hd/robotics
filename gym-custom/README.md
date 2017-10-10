# Code Explaination

## Environment Code
In folder `gym_custom/envs`:


### gazebo_env.py
This is parent file which others environment file will be implement based on.
It includes some functions that were implement and some abstract functions:

#### __init__:
Open server of **ROS**, init **Gazebo** node and launch environment.

#### _render:
Control **Gazebo client** (display problems).

#### _close:
Close **Gazebo** and **ROS**.

### Environment files
These define variables of the environment we use include state, reward, action, destination....

#### __init__:
Define services, variables we will use.

#### _step:
Get an action as input and return state, reward, done (episode finished or not) and other information (optional)
States, rewards and actions are defined here.

#### _reset:
Make environment return to default.

NOTE: This folder includes some environments:
`turtlebot_around_camera.py`: states are image/depth_image and 5 actions.

`turtlebot_follows_walls.py`: uses distance (got by laser) to be states and includes 8 actions.

`turtlebot_target_camera.py`: uses image to be states with some positions have larger rewards are target and includes 5 actions.

`turtlebot_target_camera_laser.py`: same as `turtlebot_target_camera.py`.

`turtle_world.py` and `turtle_world_position.py`: simply uses position as states with 3 actions.



## Algorithm Code
In folder `gym_custom/src/`:
All codes use **Q-learning**.

Example with files implement **Q-learning** with **Neural Network**:
### Main idea
Use a replay memory to store old steps that robot took.
Each time, take a batch sample from replay memory to train through Neural Netword to update weights of net.

### Implement
We need 2 class: Q-network (implement our net) and ExperienceReplay (store old taken steps).
In each step of each episode:
- Get action by
	- Random if total steps robot took less than the number of pre-train or the posibility choosing action is less than epsilon.
	- Predict by the network (max of Q-values we get from the network if input is current state).

- Do action and take back reward from environment.

- Store it in replay memory (includes `old_observation, action, reward, done and new_observation`).

- If total taken step is more than number of pre-train:
	- Get a sample from memory
	- Train with our network:
		- Get Q-values if input is old observation (Ex: Q-all).
		- Get Q-values if input is new observation (Ex: Q1).
		- Update Q-values of old observation with action by formula:
			**Q-all[action] = reward + gamma * max(Q1)** 


NOTE: 
In those file, we use 2 replay memory. The first is replay memory of each episode and the second is replay memory of robot which is updated after each episode by store all experiences from the first.

In this folder, we have some files which use different environment files from first part.
