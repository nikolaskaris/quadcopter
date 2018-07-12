import numpy as np
from physics_sim import PhysicsSim
from my_functions import Sigmoid

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=20., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.start_pos = self.sim.pose[:3]
        self.action_repeat = 3

        # state made of current position, velocity and angular velocity
        self.state_size = self.action_repeat * (6 + 3 + 3)
        self.action_low = 300    # default 0
        self.action_high = 1000 # default 900
        self.action_size = 4   # default 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, old_angular_v, old_v): 
        """Uses current pose of sim to return reward."""
        
        # Squared distance
        squared_x_diff = abs(self.sim.pose[0] - np.float32(self.target_pos[0]))**2
        squared_y_diff = abs(self.sim.pose[1] - np.float32(self.target_pos[1]))**2
        squared_z_diff = abs(self.sim.pose[2] - np.float32(self.target_pos[2]))**2
        
        # distance_from_target = Sigmoid(sum(abs(self.sim.pose[:3] - np.float32(self.target_pos))) / 3)
        # distance_from_target = Sigmoid(np.sqrt(squared_x_diff + squared_y_diff + squared_z_diff))
        distance_from_target = Sigmoid(sum(abs(self.sim.pose[:3] - np.float32(self.target_pos))) / 3)

        # ABS distance
        # x_distance_from_target = abs(self.sim.pose[0] - np.float32(self.target_pos[0]))
        # y_distance_from_target = abs(self.sim.pose[1] - np.float32(self.target_pos[1]))
        # z_distance_from_target = Sigmoid(abs(self.sim.pose[2] - np.float32(self.target_pos[2])))
        
        # punish large deltas in euler angles and velocity to produce smooth flight
        # euler_change = Sigmoid(sum(abs(old_angular_v - self.sim.angular_v)))
        # velocity_change = Sigmoid(sum(abs(old_v - self.sim.v)))
        
        # Reward less and less distance from target
        reward = 1.0 - distance_from_target
        
        # Punish large changes in angular velocity and velocity
        # reward -= (euler_change + velocity_change)

        return reward

#         """Uses current pose of sim to return reward."""
#         reward = 0
#         penalty = 0
        
#         current_position = self.sim.pose[:3]
#         # penalty for euler angles, we want the takeoff to be stable
#         penalty += 2*abs(self.sim.pose[3:6]).sum()
        
#         reward -= penalty
        
#         # Target height for takeoff
#         target_z = self.target_pos[2]  # target Z
#         current_z = self.sim.pose[2]   # current Z
        
        
#         reward += -min(abs(target_z - current_z), 20.0)  # reward = zero for matching target z
#         if current_z >= target_z:  # agent has crossed the target height
#             reward += 10.0  # bonus reward
#             done = True
            
        

        
#         # link velocity to residual distance
#         # penalty += abs(abs(current_position-self.target_pos).sum() - abs(self.sim.v).sum())
        
       

#         distance = np.sqrt(squared_x_diff + squared_y_diff + squared_z_diff)
        
#         # penalty for distance from target
#         penalty += squared_z_diff**0.5
        
        
#         # extra reward for flying near the target 
#         reward += 1 / distance**0.5
        
#         reward =- penalty
#         return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            old_pose = self.sim.pose
            old_angular_v = self.sim.angular_v
            old_v = self.sim.v
            
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(old_angular_v, old_v)
            pose_all.append(self.current_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def current_state(self):
        """The state contains information about current position, velocity and angular velocity"""
        state = np.concatenate([np.array(self.sim.pose), np.array(self.sim.v), np.array(self.sim.angular_v)])
        return state

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.current_state()] * self.action_repeat)
        return state
