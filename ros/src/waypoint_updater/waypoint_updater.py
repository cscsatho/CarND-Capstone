#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import math
import numpy as np

#### Debugging code
#### from multiprocessing import Lock
#### 
#### class DbgLogger:
####     '''
####     Debugging utility it will write messages to a log file for analysis.
####     Writes are locked to ensure thread safety
####     Encapsulating it in a class to ensure the file is closed in the destructor.
####     The log file will be stored typically at ~/.ros/waypt_updt.log
####     '''
####     def __init__(self):
####         self.dbgfile = open("waypt_updt.log", "w")
####         self.mutex = Lock()
#### 
####     def write(self, msg):
####         with self.mutex:
####             self.dbgfile.write("{}\n".format(msg))
#### 
####     def __del__(self):
####         self.dbgfile.close()
####        
#### # Logger instance
#### dbgLogger = DbgLogger()
#### def dbgLog(msg):
####     '''
####     Simple wrapper to write messages uses the logger instance
####     '''
####     fmtmsg = "WPUDT:{}".format(msg)
####     print(fmtmsg)
####     rospy.loginfo(fmtmsg)
####     dbgLogger.write(fmtmsg)
#### 


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Add subscribers for /current_pose, base_waypoints, traffic_waypoint
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # TODO: Add a subscriber for /obstacle_waypoint below

        # Add a publisher for /final_waypoints 
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add other member variables needed below
        self.waypoints_2d = None
        self.base_waypoints = None
        self.pose = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        #### dbgLog("stopline_wp_idx: {}".format(self.stopline_wp_idx))
        self.loop()


    def loop(self):
        '''
        Ensures that the publishing frequency is 50Hz.
        Closest waypoints are found by querying the KDTree.
        '''
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_2d and self.waypoint_tree is not None:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                #### dbgLog("closest_waypoint_idx: {}".format(closest_waypoint_idx))
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
   
    def get_closest_waypoint_idx(self):
        '''
        Queries the waypoint tree and returns the index of the closest waypoint ahead of the car.
        '''
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        closest_coord = self.waypoints_2d[closest_idx]
        # If closest is behind us, need to find the closest point ahead.
        # Use dot product to determine if the closest is behind, if so return the next
        prev_coord = self.waypoints_2d[closest_idx - 1]

        #Hyperplane through closest_coords, to find closest waypoint that is AHEAD of the car
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            # Closest is behind, find the next
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx
 
    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            #No traffic light detected
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            #Go two waypoints to ensure vehicle stops at the line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2.0 * MAX_DECEL * dist)
            if vel < 1:
                vel = 0.0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        '''
        Callback for /base_waypoints
        This sets the base_waypoints to the input waypoints and initializes a KDTree so that 
        nearest neighbor queries can be answered efficiently.
        '''
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        #### dbgLog("traffic_cb: {}".format(msg.data))
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
