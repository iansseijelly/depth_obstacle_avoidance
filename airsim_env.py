import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np
import airsim
from disparity import compute_disparity_py

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Calculate the distance from a point (px, py) to a line segment (ax, ay)-(bx, by) in 2D."""
    # Vector AB
    ABx = bx - ax
    ABy = by - ay
    
    # Vector AP
    APx = px - ax
    APy = py - ay
    
    # Vector BP
    BPx = px - bx
    BPy = py - by
    
    # Calculate the projection of AP onto AB to find the nearest point on the segment
    AB_AB = ABx * ABx + ABy * ABy
    AB_AP = ABx * APx + ABy * APy
    
    t = AB_AP / AB_AB
    
    if t < 0.0:
        # Closest point is A
        nearest_x, nearest_y = ax, ay
    elif t > 1.0:
        # Closest point is B
        nearest_x, nearest_y = bx, by
    else:
        # Projection falls on the segment
        nearest_x = ax + t * ABx
        nearest_y = ay + t * ABy
    
    # Distance from P to the nearest point on the segment
    dx = px - nearest_x
    dy = py - nearest_y
    
    return np.sqrt(dx * dx + dy * dy)

def min_distance_to_trajectory(point, trajectory):
    """Find the minimum distance from a point to any segment in a trajectory."""
    min_distance = float('inf')
    px, py = point
    
    num_points = len(trajectory)
    
    for i in range(num_points - 1):
        # Segment from point i to point i+1
        ax, ay = trajectory[i][:2]
        bx, by = trajectory[i+1][:2]
        
        # Calculate the distance from the point to the segment
        distance = point_to_segment_distance(px, py, ax, ay, bx, by)
        
        # Update the minimum distance
        min_distance = min(min_distance, distance)
    
    return min_distance


def transform_trajectory(trajectory, scalar):
    """Transform the trajectory so that the first point is the origin and scale all values."""
    # Get the first point
    first_point = trajectory[0]
    x0, y0, z0 = first_point[:3]
    
    # Transform the trajectory
    transformed_trajectory = []
    for point in trajectory:
        x, y, z = point
        transformed_x = (x - x0) * scalar
        transformed_y = (y - y0) * scalar
        transformed_z = (z - z0) * scalar
        transformed_trajectory.append((transformed_x, transformed_y, transformed_z))
    
    return transformed_trajectory


def compute_disparity(left, right, min_disparity, max_disparity, half_block_size):
    height, width = left.shape
    disparity = np.zeros((height-(max_disparity-min_disparity)-2*half_block_size, width-2*half_block_size), dtype=np.int8)

    # Pad the images to handle the block size
    pad_size = half_block_size + max_disparity
    left_padded = np.pad(left, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    right_padded = np.pad(right, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

    for i in range(half_block_size, height - half_block_size):
        for j in range(half_block_size, width - half_block_size):
            min_SAD = np.iinfo(np.int32).max
            best_offset = 0

            # Extract the left block
            left_block = left_padded[i:i + 2 * half_block_size + 1, j:j + 2 * half_block_size + 1]

            for offset in range(min_disparity, max_disparity):
                # Extract the right block with the offset
                right_block = right_padded[i:i + 2 * half_block_size + 1, j + offset:j + offset + 2 * half_block_size + 1]

                # Compute the SAD for the current offset
                SAD = np.sum(np.abs(left_block - right_block))

                if SAD < min_SAD:
                    min_SAD = SAD
                    best_offset = offset

            disparity[i-half_block_size, j-half_block_size] = best_offset
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity

def max_pooling(image, pool_size):
    pooled_image = cv2.resize(image, (image.shape[1] // pool_size, image.shape[0] // pool_size), interpolation=cv2.INTER_NEAREST)
    pooled_image = np.maximum.reduceat(np.maximum.reduceat(pooled_image, np.arange(0, pooled_image.shape[0], pool_size), axis=0), np.arange(0, pooled_image.shape[1], pool_size), axis=1)
    return pooled_image

def simple_max_pooling(image, pool_size):
    # Ensure that the image dimensions are divisible by pool_size
    height, width = image.shape
    new_height = height // pool_size
    new_width = width // pool_size

    # Create an empty array for the pooled image
    pooled_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Iterate over each pool_size x pool_size tile
    for i in range(new_height):
        for j in range(new_width):
            tile = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            pooled_image[i, j] = np.max(tile)
    
    return pooled_image

class AirSimEnv(gym.Env):

    def __init__(self, *args, vehicle="drone", airsim_ip="localhost", airsim_port=8187, random_yaw=None, random_pos=None, trajectory=None, destination=None, **kwargs):
        super(AirSimEnv, self ).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example: Action is a discrete choice of 3 possibilities


        # TODO Load from config:
        self.image_dim = 256
        self.search_range = 32
        self.block_size = 8
        self.stereo_width = self.image_dim - self.search_range - self.block_size
        self.stereo_height = self.image_dim - self.block_size
        
        self.random_yaw = random_yaw
        self.random_pos = random_pos
        # self.image_dim = 64
        
        self.trajectory = None
        if trajectory is not None:
            self.trajectory = transform_trajectory(trajectory, 0.01)
            
        self.destination = destination

        # Action
        # Interpretation of targets:
        # [z, xvel, yvel, yawrate]
        self.action_space = spaces.Dict({
            # TODO check if this actually clips
            "targets": spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32),
            "active": spaces.Discrete(2)
        })

        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3), dtype=np.uint8),
            "imu": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "depths": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "disp": spaces.Box(low=0, high=255, shape=(self.stereo_height, self.stereo_width), dtype=np.uint8),
            "sobel": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim), dtype=np.uint8),
        })
        print(f"obs space: {self.observation_space['camera'].shape}")

        # Initialize the AirSim simulation
        print("Connecting to AirSim server")

        self.vehicle = vehicle
        if self.vehicle == "drone":
            self.client = airsim.MultirotorClient(ip=airsim_ip, port=airsim_port)
        elif self.vehicle == "car":
            self.client = airsim.CarClient(ip=airsim_ip, port=airsim_port)
        else:
            raise ValueError("AirSim only supports 'drone' and 'car' configurations.")

        # Confirm connection, enable API control to AirSim
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Ensure ensure simulator execution is pasued. Only continue on step(). 
        self.client.simPause(True)
    
    def calc_observation(self):
        get_disp = True

        # Prepare image requests
        image_requests = [
            airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, True),  # Left camera, uncompressed
            airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, True)  # Right camera, uncompressed
        ]

        # Asynchronously get images
        image_responses = self.client.simGetImages(image_requests)

        if get_disp:
            # Decode images
            left_png = cv2.imdecode(airsim.string_to_uint8_array(image_responses[0].image_data_uint8), cv2.IMREAD_COLOR)
            right_png = cv2.imdecode(airsim.string_to_uint8_array(image_responses[1].image_data_uint8), cv2.IMREAD_COLOR)

            # Resize images
            camera_observation = cv2.resize(right_png, (self.image_dim, self.image_dim))
            left_grey_img = cv2.cvtColor(left_png, cv2.COLOR_BGR2GRAY)
            right_grey_img = cv2.cvtColor(right_png, cv2.COLOR_BGR2GRAY)

            # Compute disparity (assuming you have a function compute_disparity_py)
            disp_orig = compute_disparity_py(right_grey_img, left_grey_img, 0, 32, 4)
            disp_orig = np.clip(disp_orig, 0, 31).astype(np.uint8) * 8
            # disp = cv2.resize(disp_orig, (self.image_dim, self.image_dim))
        else:
            disp = np.zeros((self.image_dim, self.image_dim), dtype=np.uint8)

        # Apply Sobel filter to the right grayscale image
        right_scaled = cv2.resize(right_grey_img, (self.image_dim * 8, self.image_dim * 8), interpolation=cv2.INTER_LINEAR)
        sobel_x = cv2.Sobel(right_scaled, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(right_scaled, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_magnitude = cv2.normalize(sobel_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        sobel_magnitude = sobel_magnitude.astype(np.uint8)
        sobel_magnitude = simple_max_pooling(sobel_magnitude, 8)  # Assuming you have a function simple_max_pooling

        # Other sensors data (e.g., IMU, depth)
        imu_observation = np.zeros(6, dtype=np.float32)
        depths_observation = np.zeros(3, dtype=np.float32)

        observation = {
            "camera": camera_observation,
            "imu": imu_observation,
            "depths": depths_observation,
            "disp": disp,
            "sobel": sobel_magnitude
        }

        cv2.imshow("AirSim Camera Feed", observation['camera'])
        cv2.imshow("AirSim Disparity Feed", observation['disp'])
        cv2.imshow("AirSim Sobel Feed", observation['sobel'])

        # cv2.imwrite("intermediate/camera.png", observation['camera'])
        # cv2.imwrite("intermediate/disparity.png", observation['disp'])
        # cv2.imwrite("intermediate/sobel.png", observation['sobel'])
        cv2.waitKey(1)  # 1 millisecond delay to allow OpenCV to process events

        return observation

    
    # def calc_observation(self):
        
    #     get_disp=True
    #     # Get camera data
    #     # rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
    #     # png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)

    #     # camera_observation = cv2.resize(png, (self.image_dim, self.image_dim))
        
    #     if get_disp:
    #         raw_left = self.client.simGetImage("front_left", airsim.ImageType.Scene)
    #     raw_right = self.client.simGetImage("front_right", airsim.ImageType.Scene)

    #     if get_disp:
    #         left_png = cv2.imdecode(airsim.string_to_uint8_array(raw_left), cv2.IMREAD_COLOR)
    #     right_png = cv2.imdecode(airsim.string_to_uint8_array(raw_right), cv2.IMREAD_COLOR)
    #     camera_observation = cv2.resize(right_png, (self.image_dim, self.image_dim))

    #     if get_disp:
    #         left_grey_img = cv2.cvtColor(left_png, cv2.COLOR_BGR2GRAY)
    #     right_grey_img = cv2.cvtColor(right_png, cv2.COLOR_BGR2GRAY)
        
    #     if get_disp:
    #         disp_orig = compute_disparity_py(right_grey_img, left_grey_img, 0, 32, 6)
    #         # print(f"disp_orig: {disp_orig}")
    #         # disp_orig = cv2.normalize(disp_orig, disp_orig, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #         disp_orig = np.clip(disp_orig, 0, 31).astype(np.uint8) * 8

    #         # The code is resizing an image using OpenCV library in Python. It resizes the image stored in
    #         # the variable `disp` to a square shape with dimensions `self.image_dim x self.image_dim`.
    #         disp = cv2.resize(disp_orig, (self.image_dim, self.image_dim))
    #     else:
    #         disp = np.zeros((self.image_dim, self.image_dim), dtype=np.uint8)

    #     # Apply Sobel filter to the right grayscale image
    #     right_scaled = cv2.resize(right_grey_img, (self.image_dim *8, self.image_dim*8), interpolation=cv2.INTER_LINEAR)
    #     sobel_x = cv2.Sobel(right_scaled, cv2.CV_64F, 1, 0, ksize=3)
    #     sobel_y = cv2.Sobel(right_scaled, cv2.CV_64F, 0, 1, ksize=3)
    #     sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    #     sobel_magnitude = cv2.normalize(sobel_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #     sobel_magnitude = sobel_magnitude.astype(np.uint8)
    #     sobel_magnitude = simple_max_pooling(sobel_magnitude, 8)
    #     # sobel_magnitude = cv2.resize(sobel_magnitude, (self.image_dim, self.image_dim))
    #     # sobel_magnitude = max_pooling(sobel_magnitude, 2)
    #     # sobel_magnitude = cv2.resize(sobel_magnitude, (self.image_dim, self.image_dim))
    #     # sobel_magnitude = cv2.resize(sobel_magnitude, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
    #     # print(f"shape: {sobel_magnitude.shape}")
        
        

    #     # Get IMU data
    #     # TODO
    #     imu_observation = np.zeros(6, dtype=np.float32)

    #     # Get depth data
    #     # TODO
    #     depths_observation = np.zeros(3, dtype=np.float32)
        

    #     observation = {
    #         "camera": camera_observation,
    #         "imu": imu_observation,
    #         "depths": depths_observation,
    #         "disp": disp,
    #         "sobel": sobel_magnitude
    #     }
    #     cv2.imshow("AirSim Camera Feed", observation['camera'])
    #     # cv2.imshow("AirSim Camera Feed",right_png)
    #     cv2.imshow("AirSim Disparity Feed", observation['disp'])
    #     # cv2.imshow("AirSim Disparity Feed", disp_orig)
    #     cv2.imshow("AirSim Sobel Feed", observation['sobel'])
    #     cv2.waitKey(1)  # 1 millisecond delay to allow OpenCV to process events
        
    #     return observation
    
    def is_pose_acceptable(self, pose):
        current_pose = self.client.simGetVehiclePose()
        # Check if the current pose is close enough to the initial desired pose
        is_close = np.linalg.norm([
            current_pose.position.x_val - pose.position.x_val,
            current_pose.position.y_val - pose.position.y_val,
            current_pose.position.z_val - pose.position.z_val
        ]) < 0.5  # Threshold of 0.1 meters
        return is_close
    
    def reset(self, seed=None, options=None):
        # Reset the simulator
        # ...
        pose = self.client.simGetVehiclePose()

        valid = False

        while not valid:
            # TODO Replace with Config
            #self.initial_x = 10
            self.initial_x = 0
            self.initial_y = 0
            # self.airsim_step = 6
            self.airsim_step = 3

            # TODO Replace with Config
            try:
                f = open('angle.txt', 'r')
                yaw = float(f.readline()) * np.pi / 180
                pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
            except:
                # pose.orientation = airsim.utils.to_quaternion(0,0, np.pi)
                if self.random_yaw is not None:
                    pose.orientation = airsim.utils.to_quaternion(0,0, np.random.uniform(-np.pi, np.pi))
                else:
                    pose.orientation = airsim.utils.to_quaternion(0,0, -np.pi/2)
            
            #reset velocity

            self.client.armDisarm(False)
            pose.position.x_val = self.initial_x
            pose.position.y_val = self.initial_y
            pose.position.y_val = -1.5
            if self.random_pos is not None:
                pose.position.x_val = np.random.uniform(-20, 20)
                pose.position.y_val = np.random.uniform(-20, 20)
            if self.vehicle == "drone":
                pose.position.z_val = -1.5
            # pose.position.z_val -= 30
            # self.client.simSetVehiclePose(pose, ignore_collision=True)
            # self.client.simContinueForFrames(10)
            # pose.position.z_val += 30

            # self.client.simSetVehiclePose(pose, ignore_collision=True)
            # self.client.simContinueForFrames(10)

            # self.client.simSetVehiclePose(pose, ignore_collision=True)
            # self.client.simContinueForFrames(1)
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.simContinueForFrames(1)
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            self.client.simContinueForFrames(1)
        
            # while not self.is_pose_acceptable(pose):
            #     self.client.reset()
            #     self.client.simSetVehiclePose(pose, ignore_collision=True)
            #     self.client.simContinueForFrames(10)

            self.client.armDisarm(True)
            self.client.simContinueForFrames(1)
            # self.control.targets['running'] = True

            #TODO Add Logging

            initial_observation = self.calc_observation()

            # valid of no collisions and depth sensor reading is under 10
            collision = self.client.simGetCollisionInfo().has_collided
            distance_sensor_data = self.client.getDistanceSensorData(distance_sensor_name='DistanceD')
            raw_height = distance_sensor_data.distance  # Raw height from sensor
            roll, pitch, yaw = airsim.to_eularian_angles(pose.orientation)
            adjusted_height = raw_height * np.cos(roll) * np.cos(pitch)
            height = adjusted_height
            # print("height: ", height)
            if not collision and height < 10:
                valid = True
            
            if valid:
                # Return the initial observation
                return initial_observation, {}
    
    def step(self, action):
        # Apply the action to the simulator
        # ...
        # Get state data
        pose = self.client.simGetVehiclePose()
        x = pose.position.x_val 
        y = pose.position.y_val 

        # Interpretation of targets:
        # [z, xvel, yvel, yawrate]
        print(f"action: {action}")
        z = action['targets'][0]
        xvel = action['targets'][1]
        yvel = action['targets'][2]
        yawrate = action['targets'][3]
        # print(f"action: {action}")
        height = 0
        x_vel = 0

        # print("dist: ", dist)

        if action['active']:
            kin = self.client.simGetGroundTruthKinematics()
            distance_sensor_data = self.client.getDistanceSensorData(distance_sensor_name='DistanceD')
            raw_height = -distance_sensor_data.distance  # Raw height from sensor
            roll, pitch, yaw = airsim.to_eularian_angles(kin.orientation)

            # Adjust height based on pitch and roll
            adjusted_height = raw_height * np.cos(roll) * np.cos(pitch)
            height = adjusted_height

            # height = kin.position.z_val
            # print(f"height: {height}")
            yaw = airsim.to_eularian_angles(kin.orientation)[2] * 180/np.pi
            x_vel = kin.linear_velocity.x_val * np.cos(np.radians(yaw)) + kin.linear_velocity.y_val * np.sin(np.radians(yaw))
            y_vel = -kin.linear_velocity.x_val * np.sin(np.radians(yaw)) + kin.linear_velocity.y_val * np.cos(np.radians(yaw))
            z_vel = kin.linear_velocity.z_val
            #throttle = 0.6 + 0.1*(height - z)
            throttle = 0.6 + 0.2*(height - z)
            if height > z and z_vel > 0:
                throttle += z_vel*0.2
            if height < z and z_vel < 0:
                throttle += z_vel*0.2
            pitch = (xvel - x_vel) *  0.1
            roll = (yvel - y_vel) * 0.2 
            yawrate = yawrate * 0.5
            if (yawrate > 0 and yvel > 0) or (yawrate < 0 and yvel < 0):
                yawrate = 0

            # print(f"throttle: {throttle}, pitch: {pitch}, roll: {roll}, yawrate: {yawrate}")
            self.client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yawrate, throttle ,0.3) 
        self.client.simContinueForTime(self.airsim_step/100)

        # Continue simulation until simulation finishes simulating the current timestep
        while True:
            if (self.client.simIsPause()):
                break

        # Get the new observation, reward, and check if the episode is done
        # ...
        
        collision = self.client.simGetCollisionInfo().has_collided


        observation = self.calc_observation()
        # reward = 2

        # # if (x < 0):
        # #     reward -= x * 0.1
        # # if (x > 0):
        # #     reward -= x * 10
        # done        = False
        
        # # if height < -2 or height > 2:
        # #     reward -= abs(height) * 10
        # if height < -10 or height > 10:
        #     reward -= abs(height) * 10
            
        # # reward += x_vel

        # if collision:
        #     done = True
        #     reward = -100
            
        # if self.trajectory is not None:
        #     min_distance = min_distance_to_trajectory((x, y), self.trajectory)
        #     if min_distance > 0: 
        #         reward -= min_distance * 0.5
        #     if min_distance > 4:
        #         done = True
        #         reward -= 100
            # Reward function
        reward = 0

        # Penalize for collisions
        if collision:
            done = True
            reward -= 100
        else:
            done = False

        # reward for getting to destination
        if self.destination is not None:
            dist = np.linalg.norm([x - self.destination[0], y - self.destination[1]])
            if dist < 3:
                reward += 100
                done = True

        # Penalize for being too high or too low
        if height < -12 or height > 12:
            reward -= abs(height) * 10

        # Reward for moving forward
        reward += x_vel * 0.75

        # Penalize for distance from the trajectory
        if self.trajectory is not None:
            min_distance = min_distance_to_trajectory((x, y), self.trajectory)
            reward -= min_distance * 1
            if min_distance > 4:
                done = True
                reward -= 100
            

        # Optionally, you can provide additional info (it's not used for training)
        info = {}
        print(f"reward: {reward}")

        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        observation = self.calc_observation()
        camera_image = observation['disp']
        return camera_image

    def close(self):
        # Close resources
        # ...
        pass

    def configure_logs(self):
        pass
