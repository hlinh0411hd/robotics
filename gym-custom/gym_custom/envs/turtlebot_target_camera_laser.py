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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class TurtlebotTargetCameraLaserEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TurtlebotWorldCamera_v0 .launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        
        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def discretize_observation(self,data,new_ranges):
        des = [[-6, 2], [-6, -4], [7, -6.5], [4, -2.5], [7, 2.5]]
        discretized_ranges = []
        min_range = 0.2
        done = False
        destination = False
        parts = [1, len(data.ranges)/5, 2*len(data.ranges)/5, 3*len(data.ranges)/5, 4*len(data.ranges)/5] #5 parts
        """
        for i in range(len(data.ranges)):
            if data.ranges[i] < 0.1:
               ranges = list(data.ranges)
               ranges[i] = 100
               data.ranges = tuple(ranges)
        """
        discretized_ranges.append(min(data.ranges[parts[0]:parts[1]]))
        discretized_ranges.append(min(data.ranges[parts[1]:parts[2]]))
        discretized_ranges.append(min(data.ranges[parts[2]:parts[3]]))
        discretized_ranges.append(min(data.ranges[parts[3]:parts[4]]))
        discretized_ranges.append(min(data.ranges[parts[4]:len(data.ranges)-1]))

        pose = None
        while pose is None:
            try:
                pose = self.model_state('mobile_base','').pose
            except:
                print("Pose error")
        if (min(discretized_ranges) <= min_range):
            done = True
        if (abs(pose.position.x - 7) <= min_range and abs(pose.position.y - 2.5) <= min_range):
            done = True
        for i in des:
           if (abs(pose.position.x - i[0]) <= min_range and abs(pose.position.y - i[1]) <= min_range):
               destination = True
        """
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0.1):
                print data.ranges[i]
                done = True
        """

        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h/2,w/2,0]==178 and cv_image[h/2,w/2,1]==178 and cv_image[h/2,w/2,2]==178):
                    success = True
                else:
                    print("/camera/rgb/image_raw ERROR, retrying")
            except:
                print("Image error")
        cv_image = cv2.resize(cv_image, (160, 120))
        return cv_image, destination, discretized_ranges,done


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
                pass


        while (self.physics_properties().pause == False):
            try:
                #resp_pause = pause.call()
                self.pause()
            except rospy.ServiceException, e:
                print ("/gazebo/pause_physics service call failed")
        #print(len(data.ranges))
        #print(data.ranges)
        state, destination, distance, done = self.discretize_observation(data,5)
        #print state
        if destination == True:
            reward = 1000
        elif done:
            reward = -200
        else:
            reward = 0

        return state, reward, done, {}

    def _reset(self):

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

        state, destination, distance, done = self.discretize_observation(data,5) 

        return state
