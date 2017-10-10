import gym
import rospy
import roslaunch
import time
import math
import numpy as np
import cv2

from gym import utils, spaces
from gym_custom.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class TurtlebotAroundCameraEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TurtlebotWorldCamera_v1.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        
        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)

        self.action_space = spaces.Discrete(5)
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def discretize_observation(self,data):
        discretized_ranges = []
        min_range = 0.2
        done = False
        parts = [1, len(data.ranges)/6, len(data.ranges)/3, len(data.ranges)*2/3] #4 parts
        discretized_ranges.append(round(min(data.ranges[parts[0]:parts[2]]),1))
        #discretized_ranges.append(round(min(data.ranges[parts[1]:parts[2]]),1))
        discretized_ranges.append(round(min(data.ranges[parts[2]:parts[3]]),1))
        discretized_ranges.append(round(min(data.ranges[parts[3]:len(data.ranges)-1]),1))

        if (min(discretized_ranges) <= min_range):
            done = True

        image_data = None
        cv_image = None
        n = 0
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "32FC1")
            except:
                n += 1
                if n == 10:
                    print "Depth error"
                    state = []
                    done = True
                    return state, done
        cv_image = np.array(cv_image, dtype=np.float32)
        cv2.normalize(cv_image, cv_image, 0, 1, cv2.NORM_MINMAX)
        cv_image = cv2.resize(cv_image, (160, 120))
        for i in range(120):
            for j in range(160):
                if np.isnan(cv_image[i][j]):
                    cv_image[i][j] = 0
                elif np.isinf(cv_image[i][j]):
                    cv_image[i][j] = 1
        """
        image_data = None
        cv_image = None
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                print("Image error")
                pass
        cv_image = cv2.resize(cv_image, (160, 120))"""
        return cv_image, done


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        while (self.physics_properties().pause == True):
            try:
                self.unpause()
            except rospy.ServiceException, e:
                print ("/gazebo/unpause_physics service call failed")

        #print action
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.3
        if action == 0: #Turn 90
            vel_cmd.angular.z = math.pi/2
        elif action == 1: #Turn 45
            vel_cmd.angular.z = math.pi/4
        elif action == 2: #Turn 0
            vel_cmd.angular.z = 0
        elif action == 3: #Turn -45
            vel_cmd.angular.z = -math.pi/4
        elif action == 4: #Turn -90
            vel_cmd.angular.z = -math.pi/2
        self.vel_pub.publish(vel_cmd)
        """
        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 1
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -1
            self.vel_pub.publish(vel_cmd)
        """
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                state = []
                done = True
                reward = 0
                return state, reward, done, {}


        while (self.physics_properties().pause == False):
            try:
                #resp_pause = pause.call()
                self.pause()
            except rospy.ServiceException, e:
                print ("/gazebo/pause_physics service call failed")
        #print(len(data.ranges))
        #print(data.ranges)
        state, done = self.discretize_observation(data)
        #print state
        if done:
           reward = -200
        else:
           reward = 1

        return state, reward, done, {}

    def _reset(self):

        cv2.destroyAllWindows()
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException, e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        while (self.physics_properties().pause == True):
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except rospy.ServiceException, e:
                print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        while (self.physics_properties().pause == False):
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                #resp_pause = pause.call()
                self.pause()
            except rospy.ServiceException, e:
                print ("/gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data) 

        return state
