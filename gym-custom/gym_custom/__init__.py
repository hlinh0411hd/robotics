from gym.envs.registration import register

register(
    id='BlackArea-v1',
    entry_point='gym_custom.envs.black_area:BlackAreaEnvV1',
    max_episode_steps=200,
)

register(
    id='TurtlebotWorld-v0',
    entry_point='gym_custom.envs.turtle_world:TurtlebotWorldEnv',
)

register(
    id='TurtlebotWorldPosition-v0',
    entry_point='gym_custom.envs.turtle_world_position:TurtlebotWorldPositionEnv',
)

register(
    id='TurtlebotFollowsWalls-v0',
    entry_point='gym_custom.envs.turtlebot_follows_walls:TurtlebotFollowsWallsEnv',
)

register(
    id='TurtlebotTargetCamera-v0',
    entry_point='gym_custom.envs.turtlebot_target_camera:TurtlebotTargetCameraEnv',
)

register(
    id='TurtlebotAroundCamera-v0',
    entry_point='gym_custom.envs.turtlebot_around_camera:TurtlebotAroundCameraEnv',
)

