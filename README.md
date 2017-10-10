### Introduction

- Theses code were built based on [erlerobotics/gym](http://erlerobotics.com/docs/Simulation/Gym/) and using its model turtlebot
- These implemented Q-learning with local location or image / depth image to train the model.
- It includes some example environments and programmes

### Installation

- Install [erlerobotics/gym](http://erlerobotics.com/docs/Simulation/Gym/) and run test with turtlebot (to include some PATHs into file ~/.bashrc)

- In terminal:
```bash
git clone https://gitlab.com/UET-SmartRobots/Environment-learn.git
cd Environment-learn/gym-custom
sudo -H pip install .e
```

- NOTE: After installing, needs to replace world links in model files in folder	`gym_custom/envs/assets/launch`

### Running

- Sure that running:
```bash
source gym-gazebo/gym_gazebo/envs/installation/catkin_ws/devel/setup.bash
```

- Now running any examples in `gym-custom/src`

### Others
- Some models/worlds in folder `Ex-model`
