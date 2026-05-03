import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

from enum import Enum

class Lap(Enum):
    FIRST_LAP = 1
    FAST_LAP = 2

class Mode(Enum):
    TAKE_OFF = 3
    GO_TO_SEARCH_AREA = 4
    SEARCH_GATE = 5
    APPROACH_GATE = 6
    TAKE_SECOND_PHOTO = 7
    FLY_THROUGH_GATE = 8
    GO_HOME = 9
    READY_TO_EXECUTE_TRAJECTORY = 10
    EXECUTE_TRAJECTORY = 11
    LAND = 12

# Pink gates
gates_r_value = 211
gates_g_value = 144
gates_b_value = 222
GATE_HEIGHT = 0.4 # In meters 

# Rotation amounts
search_gate_rotation = 0.15
search_gate_translation = 0.4 # TODO - change back to 0.25 for final

# Approach distance
APPROACH_DIST = 0.8  # metres in front of gate to take measurement

# Fly through distance - how far forward to go after the gate center when flying through
FLY_THROUGH_DIST = 0.1

# Thresholds
eps = 0.02 # Used to check if yaw is correct - Approx 1 degree
pos_eps = 0.05 # Allow 5cm variation in approaching gate
edge_threshold = 10  # Used to check if the detected gate is at the edge (pixels tolerance from edge)

# Pause durations
PAUSE_AT_MEASUREMENT_POS = 1.5  # TODO - change back to 3.0 for final seconds — wait before taking 2nd photo
PAUSE_AT_GATE_CENTER     = 1.0  # TODO - change back to 1.5 for final seconds — wait after reaching gate center

# Camera constants
CAM_FOV = 1.5  # radians
CAM_WIDTH = 300  # pixels
CAM_HEIGHT = 300
CAM_OFFSET_BODY = np.array([0.03, 0, 0.01])  # camera offset from drone body
F_PIXELS = CAM_WIDTH / (2 * np.tan(CAM_FOV / 2))  # focal length in pixels (~161)

# Rotation from camera frame to body frame (from appendix: zcam=xdrone, xcam=-ydrone, ycam=-zdrone)
R_CAM_TO_BODY = np.array([
    [ 0,  0,  1],   # xbody =  zcam
    [-1,  0,  0],   # ybody = -xcam
    [ 0, -1,  0]    # zbody = -ycam
])

# Arena info
ARENA_CENTER = np.array([4.0, 4.0])
HOME_POSITION = np.array([1.0, 4.0, 0.7])
LAND_POSITION = np.array([1.0, 4.0, 0.1])
SEARCH_RADIUS = 4  # meters from center
SEARCH_HEIGHT = 1.35  # meters

# Find the gate search positions:
GATE_SEARCH_POSITIONS = {
    0: {'pos': np.array([4.0 + SEARCH_RADIUS*(-np.cos(np.radians(30))),
                         4.0 + SEARCH_RADIUS*(-np.sin(np.radians(30))),
                         SEARCH_HEIGHT]),
        'yaw': np.radians((30 + 180 + 90) % 360)},   # sector 2 center = 30°

    1: {'pos': np.array([4.0 + SEARCH_RADIUS*(-np.cos(np.radians(90))),
                         4.0 + SEARCH_RADIUS*(-np.sin(np.radians(90))),
                         SEARCH_HEIGHT]),
        'yaw': np.radians((90 + 180 + 90) % 360)},   # sector 4 center = 90°

    2: {'pos': np.array([4.0 + SEARCH_RADIUS*(-np.cos(np.radians(150))),
                         4.0 + SEARCH_RADIUS*(-np.sin(np.radians(150))),
                         SEARCH_HEIGHT]),
        'yaw': np.radians((150 + 180 + 90) % 360)},  # sector 6 center = 150°

    3: {'pos': np.array([4.0 + SEARCH_RADIUS*(-np.cos(np.radians(210))),
                         4.0 + SEARCH_RADIUS*(-np.sin(np.radians(210))),
                         SEARCH_HEIGHT]),
        'yaw': np.radians((210 + 180 + 90) % 360)},  # sector 8 center = 210°

    4: {'pos': np.array([4.0 + SEARCH_RADIUS*(-np.cos(np.radians(270))),
                         4.0 + SEARCH_RADIUS*(-np.sin(np.radians(270))),
                         SEARCH_HEIGHT]),
        'yaw': np.radians((270 + 180 + 90) % 360)},  # sector 10 center = 270°
}

# Add inward direction vectors to GATE_SEARCH_POSITIONS:
for _gate_idx, _entry in GATE_SEARCH_POSITIONS.items():
    _pos_xy = _entry['pos'][:2]
    _to_center = ARENA_CENTER - _pos_xy
    _entry['inward_dir'] = _to_center / np.linalg.norm(_to_center)  # unit vector toward arena center

# Trajectory constants
TRAJ_SPEED = 5.0          
MAX_VELOCITY = 7.0        
MAX_ACCELERATION = 7.0    
TRAJ_DT = 0.02           

# --- Adaptive Lookahead Parameters ---
DIST_NEAR = 0.6   
DIST_FAR = 2.0    
LOOKAHEAD_MIN = 0.4 
LOOKAHEAD_MAX = 1.8 

# --- Curvature Check Parameters ---
CURVATURE_TIME_AHEAD_1 = 1.2    # Seconds ahead to sample the first trajectory vector
CURVATURE_TIME_AHEAD_2 = 1.5    # Seconds ahead to sample the second trajectory vector
MIN_VECTOR_NORM = 0.01          # Minimum distance (m) required to safely calculate angles
MAX_TURN_ANGLE_RAD = np.pi / 2.0 # 90 degrees in radians

# --- Tracking & Gate Passage Parameters ---
GATE_PASS_ENTRY_DIST = 0.2      # Distance (m) to enter the gate proximity bubble
GATE_PASS_EXIT_DIST = 0.5       # Distance (m) to exit the bubble and trigger passage
TRACKER_SEARCH_WINDOW_TIME = 3.0 # Seconds of trajectory to search for the closest point
Z_CLAMP_DIST_XY = 0.8           # Lateral distance (m) to snap to the gate's exact Z height
YAW_LOOKAHEAD_TIME = 0.8        # Seconds ahead on the trajectory to point the camera/yaw
TRAJ_COMPLETE_DIST = 0.2        # Distance (m) from final waypoint to trigger GO_HOME

class MyAssignment:
    def __init__(self, ):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.mode = Mode.TAKE_OFF
        self.has_taken_off = False
        self.gate_corners_dict = {}         # key: gate_index → value: corners - for later laps use
        self.gate_center_poses_dict = {}   # key: gate_index → value: (center, yaw) - for later laps use
        self.gate_corners = [] # For first lap 
        self.gate_center_poses = [] # For first lap - list of tuples (center, yaw) in order of detection
        self.gate_detection_img = None
        self.target_gate_detection_img = None
        self.current_gate_number = 0 # Doing zero indexing

        # Variables for approach gate state:
        self.measurement_target_pos = None
        self.measurement_target_yaw = None

        # Pause tracking
        self.pause_start_time = None   # time.time() when pause began
        self.pause_duration = None     # how long to pause in seconds
        self.post_pause_mode = None    # mode to enter after pause
        self.ready_to_take_second_photo = False # Track if we've paused

        # Trajecotry variables
        self.trajectory_waypoints = []
        self.current_waypoint_index = 0
        self.current_waypoint = None
        self.current_traj_gate_number = 0

    def compute_command(self, sensor_data, camera_data, dt):
        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Default control command - TODO: remove this later 
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        if (self.current_gate_number >= 5 and self.mode != Mode.READY_TO_EXECUTE_TRAJECTORY): # Since we're zero indexed, after flying through gate 4 we are done
            self.mode = Mode.GO_HOME
            self.current_gate_number = 0 # Reset gate number for potential future laps

        # Take off command
        if (self.mode == Mode.TAKE_OFF):
            if sensor_data['z_global'] < 0.49 and not self.has_taken_off:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
                return control_command
            if not self.has_taken_off and sensor_data['z_global'] > 0.49:
                self.has_taken_off = True
                self.mode = Mode.GO_TO_SEARCH_AREA
                # print("Mode: Go to Search Area")
        # Search for gate command
        elif (self.mode == Mode.GO_TO_SEARCH_AREA):
            control_command = self.get_go_to_search_area_command(sensor_data)
        elif (self.mode == Mode.SEARCH_GATE):
            control_command = self.get_search_gate_command(camera_data, sensor_data)  
        elif (self.mode == Mode.APPROACH_GATE):
            control_command = self.get_approach_gate_command(camera_data, sensor_data) 
        elif (self.mode == Mode.FLY_THROUGH_GATE):
            control_command = self.get_fly_through_gate_command(sensor_data)
        elif (self.mode == Mode.TAKE_SECOND_PHOTO):
            control_command = self.get_capture_second_photo_command(camera_data, sensor_data)
        elif (self.mode == Mode.GO_HOME):
            control_command = self.get_go_home_command(sensor_data)
        elif (self.mode == Mode.READY_TO_EXECUTE_TRAJECTORY):
            return self.get_ready_to_execute_trajectory_command(sensor_data)
        elif self.mode == Mode.EXECUTE_TRAJECTORY:
            control_command = self.get_execute_trajectory_command(sensor_data)
        elif (self.mode == Mode.LAND):
            control_command = self.get_land_command()

        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

    def get_go_to_search_area_command(self, sensor_data):
        # This function is not currently used since we go straight to searching, but it could be used if we wanted a separate state for flying to the search area before starting to search
        target_pos = GATE_SEARCH_POSITIONS[self.current_gate_number]['pos']
        target_yaw = GATE_SEARCH_POSITIONS[self.current_gate_number]['yaw']

        if (abs(sensor_data['x_global'] - target_pos[0]) < pos_eps and
            abs(sensor_data['y_global'] - target_pos[1]) < pos_eps and
            abs(sensor_data['z_global'] - target_pos[2]) < pos_eps and
            abs((sensor_data['yaw'] - target_yaw + np.pi) % (2 * np.pi) - np.pi) < eps):
            self.mode = Mode.SEARCH_GATE
            # print("Mode: Search Gate")

        return [target_pos[0], target_pos[1], target_pos[2], target_yaw]

    def get_search_gate_command(self, camera_data, sensor_data):
        search_entry = GATE_SEARCH_POSITIONS[self.current_gate_number]
        search_yaw = search_entry['yaw']
        inward_dir = search_entry['inward_dir']

        target_gate, gate_corners = self.get_target_gate(camera_data, sensor_data)

        if target_gate is None or gate_corners is None:
            # Check if we've reached the arena center, if so reset to outer edge
            drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

            # Check if drone has passed the center by projecting its displacement from
            # the outer search position onto the inward direction. If this projection
            # exceeds SEARCH_RADIUS the drone has travelled further than the full
            # outer-to-center distance and must have passed (or crashed into) the center.
            outer_pos = search_entry['pos'][:2]
            displacement = drone_pos[:2] - outer_pos
            progress = np.dot(displacement, inward_dir)  # metres travelled toward center

            if progress >= SEARCH_RADIUS:
                # print("Search: passed arena center without finding gate, resetting to outer edge")
                outer = search_entry['pos']
                self.mode = Mode.GO_TO_SEARCH_AREA # Go back to search area state to reset position and yaw
                # print("Mode: Go to Search Area (resetting position after passing center)")
                return [outer[0], outer[1], outer[2], search_yaw]

            # Step: current position + one translation increment toward arena center
            return [
                sensor_data['x_global'] + inward_dir[0] * search_gate_translation,
                sensor_data['y_global'] + inward_dir[1] * search_gate_translation,
                search_entry['pos'][2],  # maintain search height
                search_yaw
            ]

        if self.is_target_gate_not_fully_in_FOV(camera_data, target_gate):
            # print("Search Gate: Target gate not fully in view, adjusting position")
            return self.adjust_position_for_better_FOV(camera_data, sensor_data, target_gate)

        self.target_gate_detection_img = camera_data.copy()
        self.target_gate_detection_img = cv2.polylines(
            self.target_gate_detection_img, [target_gate], isClosed=True, color=(0, 0, 255), thickness=2)

        center = gate_corners.mean(axis=0)
        gate_yaw = self.estimate_gate_orientation(gate_corners, sensor_data)
        approach_pos = self.compute_approach_position(center, gate_yaw)

        self.measurement_target_pos = approach_pos
        self.measurement_target_yaw = gate_yaw

        self.mode = Mode.APPROACH_GATE
        # print(f"Mode: Approach Gate — target={approach_pos}, gate_yaw={np.degrees(gate_yaw):.1f}°")
        return [approach_pos[0], approach_pos[1], approach_pos[2], gate_yaw]

    def get_approach_gate_command(self, camera_data, sensor_data):
        if self.measurement_target_pos is None or self.measurement_target_yaw is None:
            self.mode = Mode.GO_TO_SEARCH_AREA
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']] # TODO - or could do search area command

        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Check how far we are from our 0.8m measurement spot
        dist_to_target = np.linalg.norm(self.measurement_target_pos - drone_pos)

        # Check yaw error (wrapped to [-π, π])
        yaw_error = abs((sensor_data['yaw'] - self.measurement_target_yaw + np.pi) % (2 * np.pi) - np.pi)
        facing_gate = yaw_error < eps
        
        # If we are within 10cm of the measurement point, take the new photo
        if dist_to_target < pos_eps and facing_gate:
            self.mode = Mode.TAKE_SECOND_PHOTO
            # print("Mode: Take Second Photo - Pausing to take second photo at measurement position")
            self.start_pause(PAUSE_AT_MEASUREMENT_POS, Mode.TAKE_SECOND_PHOTO) 
        elif dist_to_target < pos_eps and not facing_gate:
            pass
            # print(f"Approach: at position but yaw not settled yet (error={np.degrees(yaw_error):.1f}°), waiting")
            
        # Either way, fly to the 0.8m mark
        return [self.measurement_target_pos[0], self.measurement_target_pos[1], self.measurement_target_pos[2], self.measurement_target_yaw]

    def get_capture_second_photo_command(self, camera_data, sensor_data): # Pause starts in approach gate
        # This function is called when we are at the measurement position and want to take a second photo after pausing
        pausing = self.is_pausing() # This will also handle the transition out of pausing when the time is up
        if pausing and not self.ready_to_take_second_photo:
            # Still pausing, hold position
            return [self.measurement_target_pos[0], self.measurement_target_pos[1],
                    self.measurement_target_pos[2], self.measurement_target_yaw]

        # Pause over, time to take the second photo and get a more accurate gate position
        target_gate, gate_corners = self.get_target_gate(camera_data, sensor_data)
        if (target_gate is None):
            # If we lost the gate, go back to searching
            self.mode = Mode.GO_TO_SEARCH_AREA
            self.ready_to_take_second_photo = False
            # print("Mode: Go to search area (gate lost at measurement pos)")
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation]
        
        if self.is_target_gate_not_fully_in_FOV(camera_data, target_gate):
            # print("Second photo: Target gate not fully in view, adjusting position")
            return self.adjust_position_for_better_FOV(camera_data, sensor_data, target_gate)
        
        # gate_corners = self.estimate_gate_position(target_gate, sensor_data)
        
        if gate_corners is not None:         
            center = gate_corners.mean(axis=0)
            gate_yaw = self.estimate_gate_orientation(gate_corners, sensor_data)

            # Store these for use in first lap so we go around the circle
            self.gate_corners.append(gate_corners)
            self.gate_center_poses.append((center, gate_yaw))

            # Store them for later laps so we know where each gate is based on its index in the circle - accounts for any gates we may have missed on the first lap and ensures we can fly through in the correct order around the circle
            gate_index = self.get_gate_index_from_position(center)
            # print(f"Detected gate center at {center}, gate index: {gate_index}")
            if gate_index is not None:
                self.gate_corners_dict[gate_index] = gate_corners
                self.gate_center_poses_dict[gate_index] = (center, gate_yaw)
            else:
                # print("Warning: Detected gate does not fall within any known gate index based on its position, it will not be stored for later laps")
                pass
            # print(f"Final High-Accuracy Gate Center: {center}")
            # print(f"Final High-Accuracy Gate Yaw: {np.degrees(gate_yaw):.1f}°")

            self.mode = Mode.FLY_THROUGH_GATE
            # print("Mode: Fly Through Gate")
            return [center[0], center[1], center[2], gate_yaw] # Don't set it as 0.1m beyond yet - we'll do that in the fly through state to ensure we are stable at the gate center before flying through        

    def get_fly_through_gate_command(self, sensor_data):
        # Get current center and yaw
        center = self.gate_center_poses[self.current_gate_number][0]
        gate_yaw = self.gate_center_poses[self.current_gate_number][1]

        # Calculate the point 0.1m beyond the gate center
        target_x, target_y, target_z = self.compute_fly_through_position(center, gate_yaw)

        # Check if we have reached the extended target point
        if (abs(sensor_data['x_global'] - target_x) < pos_eps and
            abs(sensor_data['y_global'] - target_y) < pos_eps and
            abs(sensor_data['z_global'] - target_z) < pos_eps and
            abs((sensor_data['yaw'] - gate_yaw + np.pi) % (2 * np.pi) - np.pi) < eps):
            
            # We are cleanly through the gate, move to next one
            self.current_gate_number += 1
            self.mode = Mode.GO_TO_SEARCH_AREA
            # print("Mode: Search Gate")
        
        return [target_x, target_y, target_z, gate_yaw]
    
    def get_go_home_command(self, sensor_data):
        if (abs(sensor_data['x_global'] - HOME_POSITION[0]) < pos_eps and
            abs(sensor_data['y_global'] - HOME_POSITION[1]) < pos_eps and
            abs(sensor_data['z_global'] - HOME_POSITION[2]) < pos_eps):
            self.mode = Mode.READY_TO_EXECUTE_TRAJECTORY
            self.current_traj_gate_number = 0
            # print("Mode: Ready to Execute Trajectory (inside get_go_home_command)")
            self.trajectory_waypoints = self.compute_trajectory() # TODO - could also change this to now return the compute trajectory command
        return [HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2], 0.0]
    
    def get_land_command(self):
        # print("Completed all gates, stored values are:")
        for gate_idx in range(5):
            if gate_idx in self.gate_center_poses_dict:
                center, yaw = self.gate_center_poses_dict[gate_idx]
                # print(f"Gate {gate_idx}: Center = {center}, Yaw = {yaw}")
            else:
                # print(f"Gate {gate_idx}: NOT DETECTED")
                pass
        return [LAND_POSITION[0], LAND_POSITION[1], LAND_POSITION[2], 0.0]

    def get_ready_to_execute_trajectory_command(self, sensor_data):
        # Get to the height of the first gate center before leaving the pad to save time:
        target_z = self.gate_center_poses[0][0][2] 
        # Target yaw = angle from home position toward first gate center (straight line)
        first_gate_center = self.gate_center_poses[0][0]
        dx = first_gate_center[0] - HOME_POSITION[0]
        dy = first_gate_center[1] - HOME_POSITION[1]
        target_yaw = np.arctan2(dy, dx)
        if (abs(sensor_data['x_global'] - HOME_POSITION[0]) < pos_eps and
            abs(sensor_data['y_global'] - HOME_POSITION[1]) < pos_eps and
            abs(sensor_data['z_global'] - target_z) < pos_eps and
            abs((sensor_data['yaw'] - target_yaw + np.pi) % (2 * np.pi) - np.pi) < eps):
            
            # 4. Drone has arrived. Trigger the pause instead of immediately changing mode.
            # print("Mode: Ready to Execute Trajectory - Arrived at height, pausing before execution")
            self.traj_start_time = None  
            self.current_waypoint_index = 0 # ADD THIS: Reset index for spatial tracking
            
            # I am using 1.0 second here, but feel free to adjust this pause duration
            self.mode = Mode.EXECUTE_TRAJECTORY
            # print("Mode: Execute Trajectory")
            self.start_pause(1.0, Mode.EXECUTE_TRAJECTORY) 

        return [sensor_data['x_global'], sensor_data['y_global'], target_z, target_yaw]

    def get_execute_trajectory_command(self, sensor_data):
        """
        Adaptive Pure Pursuit Tracker: Finds the closest geometric point on the 
        trajectory and looks ahead dynamically.
        """
        pausing = self.is_pausing() 
        if pausing:
            target_z = self.gate_center_poses[0][0][2] 
            first_gate_center = self.gate_center_poses[0][0]
            dx = first_gate_center[0] - HOME_POSITION[0]
            dy = first_gate_center[1] - HOME_POSITION[1]
            target_yaw = np.arctan2(dy, dx)
            return [HOME_POSITION[0], HOME_POSITION[1], target_z, target_yaw]
        
        if not hasattr(self, 'trajectory_waypoints') or self.trajectory_waypoints is None or len(self.trajectory_waypoints) == 0:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

        # --- 1. Identify the Active Gate and Check for Passage (CYLINDER PLANE-CROSSING) ---
        gate_keys = sorted(self.gate_center_poses_dict.keys())
        
        if self.current_traj_gate_number < len(gate_keys):
            active_gate_key = gate_keys[self.current_traj_gate_number]
            gate_center, _ = self.gate_center_poses_dict[active_gate_key]
            
            if self.current_traj_gate_number == 0:
                prev_point = HOME_POSITION
            else:
                prev_gate_key = gate_keys[self.current_traj_gate_number - 1]
                prev_point, _ = self.gate_center_poses_dict[prev_gate_key]
                
            approach_vec = gate_center[:2] - np.array(prev_point[:2])
            
            if np.linalg.norm(approach_vec) > MIN_VECTOR_NORM:
                approach_dir = approach_vec / np.linalg.norm(approach_vec)
            else:
                approach_dir = np.array([1.0, 0.0]) 
                
            plane_normal = np.array([approach_dir[0], approach_dir[1], 0.0])
            vec_to_drone = drone_pos - gate_center
            
            forward_dist = np.dot(vec_to_drone, plane_normal)
            dist_xy = np.linalg.norm(vec_to_drone[:2]) 
            dist_z = abs(vec_to_drone[2])              
            
            if forward_dist > 0.0 and dist_xy < 3.0 and dist_z < 4.0:
                self.current_traj_gate_number += 1

        # --- 3. Find the closest point on the trajectory (LOCALIZED VECTORIZED FIX) ---
        # Restrict the search window to 1.5s ahead and 0.5s behind. This allows catching up
        # but prevents "short-circuiting" onto overlapping future sections of the track.
        search_window_fwd = int(1.5 / TRAJ_DT) 
        search_window_bwd = int(0.5 / TRAJ_DT) 
        
        current_idx = getattr(self, 'current_waypoint_index', 0)
        start_idx = max(0, current_idx - search_window_bwd)
        end_idx = min(current_idx + search_window_fwd, len(self.trajectory_waypoints))
        
        window_points = self.trajectory_waypoints[start_idx:end_idx]
        distances = np.linalg.norm(window_points - drone_pos, axis=1)
        
        best_idx_in_window = np.argmin(distances)
        best_idx = start_idx + best_idx_in_window
                
        self.current_waypoint_index = best_idx

        # --- 2. Calculate Dynamic Lookahead: CURVATURE + GATE PROXIMITY ---
        idx_ahead_1 = min(best_idx + int(CURVATURE_TIME_AHEAD_1 / TRAJ_DT), len(self.trajectory_waypoints) - 1)
        idx_ahead_2 = min(best_idx + int(CURVATURE_TIME_AHEAD_2 / TRAJ_DT), len(self.trajectory_waypoints) - 1)
        
        vec1 = self.trajectory_waypoints[idx_ahead_1] - self.trajectory_waypoints[best_idx]
        vec2 = self.trajectory_waypoints[idx_ahead_2] - self.trajectory_waypoints[idx_ahead_1]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > MIN_VECTOR_NORM and norm2 > MIN_VECTOR_NORM:
            cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
            theta = np.arccos(cos_theta)  
        else:
            theta = 0.0
            
        turn_severity = np.clip(theta / MAX_TURN_ANGLE_RAD, 0.0, 1.0)
        curve_lookahead = LOOKAHEAD_MAX - (turn_severity * (LOOKAHEAD_MAX - LOOKAHEAD_MIN))

        dist_to_closest_gate = float('inf')
        if self.current_traj_gate_number < len(gate_keys):
            active_gate_key = gate_keys[self.current_traj_gate_number]
            gate_center, _ = self.gate_center_poses_dict[active_gate_key]
            dist_to_closest_gate = min(dist_to_closest_gate, np.linalg.norm(gate_center - drone_pos))
            
        if dist_to_closest_gate <= DIST_NEAR:
            gate_lookahead = LOOKAHEAD_MIN
        elif dist_to_closest_gate >= DIST_FAR:
            gate_lookahead = LOOKAHEAD_MAX
        else:
            ratio = (dist_to_closest_gate - DIST_NEAR) / (DIST_FAR - DIST_NEAR)
            gate_lookahead = LOOKAHEAD_MIN + ratio * (LOOKAHEAD_MAX - LOOKAHEAD_MIN)
            
        dynamic_lookahead = min(curve_lookahead, gate_lookahead)

        # --- 4. Trace forward along the path by dynamic_lookahead ---
        target_idx = best_idx
        accumulated_dist = 0.0
        
        while target_idx < len(self.trajectory_waypoints) - 1:
            step_dist = np.linalg.norm(self.trajectory_waypoints[target_idx+1] - self.trajectory_waypoints[target_idx])
            accumulated_dist += step_dist
            target_idx += 1
            
            # Anti-Reversing Trap: Make sure the target carrot is physically out in front of the drone.
            dist_from_drone = np.linalg.norm(self.trajectory_waypoints[target_idx] - drone_pos)
            if accumulated_dist >= dynamic_lookahead and dist_from_drone >= (dynamic_lookahead * 0.8):
                break

        target_pos = self.trajectory_waypoints[target_idx].copy() 

        # --- FOCUSED Z-AXIS SPLINE BELLY CLAMPING ---
        # ONLY clamp to the approaching gate. Removing the exiting gate allows instant steep dives.
        if self.current_traj_gate_number < len(gate_keys):
            active_gate_key = gate_keys[self.current_traj_gate_number]
            g_center, _ = self.gate_center_poses_dict[active_gate_key]
            
            dist_xy_to_gate = np.linalg.norm(drone_pos[:2] - g_center[:2])
            if dist_xy_to_gate < Z_CLAMP_DIST_XY:  
                target_pos[2] = g_center[2]

        # --- 5. Calculate target yaw to look slightly further ahead ---
        yaw_lookahead_idx = min(target_idx + int(YAW_LOOKAHEAD_TIME / TRAJ_DT), len(self.trajectory_waypoints) - 1)
        yaw_pos = self.trajectory_waypoints[yaw_lookahead_idx]
        dir_vec_path = yaw_pos - self.trajectory_waypoints[best_idx]
        
        if np.linalg.norm(dir_vec_path[:2]) > MIN_VECTOR_NORM:
            target_yaw = np.arctan2(dir_vec_path[1], dir_vec_path[0])
        else:
            target_yaw = sensor_data['yaw']

        # --- 6. Check if trajectory is complete ---
        dist_to_end = np.linalg.norm(self.trajectory_waypoints[-1] - drone_pos)
        if target_idx >= len(self.trajectory_waypoints) - 1 and dist_to_end < TRAJ_COMPLETE_DIST:
            self.mode = Mode.GO_HOME
            return [HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2], 0.0]

        return [target_pos[0], target_pos[1], target_pos[2], target_yaw]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # --------------- FOV adjustment helpers ---------------
    def is_target_gate_not_fully_in_FOV(self, camera_data, target_gate):
        image_height = camera_data.shape[0]
        image_width = camera_data.shape[1]

        touching_left = any(x <= edge_threshold for x, y in target_gate)
        touching_right = any(x >= image_width - edge_threshold for x, y in target_gate)
        touching_top = any(y <= edge_threshold for x, y in target_gate)
        touching_bottom = any(y >= image_height - edge_threshold for x, y in target_gate)

        return touching_left or touching_right or touching_top or touching_bottom
    
    def adjust_position_for_better_FOV(self, camera_data, sensor_data, target_gate):
        image_height = camera_data.shape[0]
        image_width = camera_data.shape[1]

        touching_left = any(x <= edge_threshold for x, y in target_gate)
        touching_right = any(x >= image_width - edge_threshold for x, y in target_gate)
        touching_top = any(y <= edge_threshold for x, y in target_gate)
        touching_bottom = any(y >= image_height - edge_threshold for x, y in target_gate)

        if touching_top and not touching_bottom:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'] + search_gate_translation, sensor_data['yaw']]
            
        elif touching_bottom and not touching_top:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'] - search_gate_translation, sensor_data['yaw']]
            
        elif touching_top and touching_bottom:
            # Calculate the backward translation components using the current yaw
            backward_dx = -np.cos(sensor_data['yaw']) * search_gate_translation
            backward_dy = -np.sin(sensor_data['yaw']) * search_gate_translation
            
            return [
                sensor_data['x_global'] + backward_dx, 
                sensor_data['y_global'] + backward_dy, 
                sensor_data['z_global'], 
                sensor_data['yaw']
            ]

        elif touching_left and not touching_right:
            # Travel towards the arena center instead of rotating
            drone_pos_xy = np.array([sensor_data['x_global'], sensor_data['y_global']])
            to_center = ARENA_CENTER - drone_pos_xy
            norm = np.linalg.norm(to_center)
            
            # Prevent division by zero if already exactly at the center
            if norm > 1e-4:
                inward_dir = to_center / norm
            else:
                inward_dir = np.array([0.0, 0.0])
                
            return [
                sensor_data['x_global'] + inward_dir[0] * search_gate_translation,
                sensor_data['y_global'] + inward_dir[1] * search_gate_translation,
                sensor_data['z_global'], 
                sensor_data['yaw']
            ]
            
        elif touching_right and not touching_left:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] - search_gate_rotation]
            
        else:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

    # --------------- Gate detection and estimation helpers ---------------
    def locate_gates(self, camera_data):
        lower, upper = self.rgb_to_hsv_bounds(r=gates_r_value, g=gates_g_value, b=gates_b_value)
        gates = self.locate_pink_area(camera_data, lower, upper)

        if gates is None:
            return None

        valid_gates = []
        for gate in gates:
            # Flatten the OpenCV contour array to a standard Nx2 array
            pts = gate.reshape(-1, 2)
            
            # 1. If it's already a perfect 4-corner shape, keep it
            if len(pts) == 4:
                valid_gates.append(pts)
                
            # 2. If noise or frame-clipping creates 5 to 10 corners, 
            # extract the 4 most extreme bounding corners.
            elif 4 < len(pts) <= 10:  
                # Top-Left has the smallest (x + y), Bottom-Right has the largest
                s = pts.sum(axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                
                # np.diff calculates (y - x). 
                # Top-Right has the smallest difference, Bottom-Left has the largest
                diff = np.diff(pts, axis=1)
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]
                
                # Reconstruct the perfect 4-corner array
                four_corner_gate = np.array([tl, tr, br, bl], dtype=np.int32)
                valid_gates.append(four_corner_gate)

        if not valid_gates:
            return None

        # Draw the successfully extracted 4-corner gates in green
        for points in valid_gates:
            self.gate_detection_img = camera_data.copy()
            self.gate_detection_img = cv2.polylines(
                self.gate_detection_img, [points], isClosed=True, color=(0, 255, 0), thickness=2
            )

        return valid_gates
    
    def get_target_gate(self, camera_data, sensor_data):
        """Returns (gate_polygon, gate_corners) or (None, None)."""
        gates = self.locate_gates(camera_data)
        if gates is None:
            return None, None

        matching_gates = []
        for gate in gates:
            if len(gate) != 4:
                continue
            corners = self.estimate_gate_position(gate, sensor_data)
            if corners is None:
                continue
            center = corners.mean(axis=0)
            gate_index = self.get_gate_index_from_position(center)
            if gate_index == self.current_gate_number:
                matching_gates.append((gate, corners))
            else:
                # print(f"Detected gate does not match current target gate index (detected index: {gate_index}, target index: {self.current_gate_number}), ignoring this detection")
                pass

        if not matching_gates:
            return None, None

        if len(matching_gates) > 1:
            drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
            matching_gates.sort(key=lambda g: np.linalg.norm(g[1].mean(axis=0) - drone_pos))

        return matching_gates[0]  # (gate_polygon, gate_corners)
    
    def estimate_gate_position(self, gate_pixels, sensor_data):
        """
        Estimates the global 3D position of all 4 gate corners.
        Calculates left and right distances independently to determine accurate yaw.
        Returns corners in a strict order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        if len(gate_pixels) != 4:
            return None

        P_cam_world = self.get_camera_position_in_world(sensor_data)
        R_body_to_world = self.get_rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
        R_cam_to_world = R_body_to_world @ R_CAM_TO_BODY

        # 1. Convert pixels to rays and group data
        points_data = []
        for px in gate_pixels:
            v_cam = self.pixel_to_direction_vector(px)
            v_world = R_cam_to_world @ v_cam
            
            norm_xy = np.hypot(v_world[0], v_world[1])
            if norm_xy < 1e-4:
                return None 
                
            slope = v_world[2] / norm_xy
            points_data.append({'px': px, 'v_world': v_world, 'norm_xy': norm_xy, 'slope': slope})

        # 2. Sort by X pixel coordinate to reliably separate Left from Right
        points_data.sort(key=lambda p: p['px'][0])
        left_side = points_data[:2]
        right_side = points_data[2:]

        # 3. Sort each side by vertical slope descending to get Top and Bottom
        left_side.sort(key=lambda p: p['slope'], reverse=True)
        right_side.sort(key=lambda p: p['slope'], reverse=True)

        tl, bl = left_side[0], left_side[1]
        tr, br = right_side[0], right_side[1]

        # 4. Calculate horizontal distance for left and right edges INDEPENDENTLY
        slope_diff_left = tl['slope'] - bl['slope']
        slope_diff_right = tr['slope'] - br['slope']

        if slope_diff_left < 1e-4 or slope_diff_right < 1e-4:
            return None 

        D_xy_left = GATE_HEIGHT / slope_diff_left
        D_xy_right = GATE_HEIGHT / slope_diff_right

        # 5. Project each corner into 3D using its respective side's distance
        corner_positions = []
        
        # We maintain a strict output order: TL, TR, BR, BL
        for corner, D_xy in [(tl, D_xy_left), (tr, D_xy_right), (br, D_xy_right), (bl, D_xy_left)]:
            v_world = corner['v_world']
            norm_xy = corner['norm_xy']
            
            cx = P_cam_world[0] + D_xy * (v_world[0] / norm_xy)
            cy = P_cam_world[1] + D_xy * (v_world[1] / norm_xy)
            cz = P_cam_world[2] + D_xy * (v_world[2] / norm_xy)
            
            corner_positions.append(np.array([cx, cy, cz]))

        return np.array(corner_positions)
    
    def estimate_gate_orientation(self, gate_corners, sensor_data):
        """
        Estimate the gate's yaw orientation by looking at the vector between the two top corners.
        Relies on the strict [TL, TR, BR, BL] ordering from estimate_gate_position.
        """
        # Because we enforced order in estimate_gate_position, we don't need unstable Z-sorting
        tl, tr = gate_corners[0], gate_corners[1]
        
        vec = tr - tl  # Vector pointing left-to-right along the top edge
        normal = np.array([-vec[1], vec[0], 0])  # Normal vector in the XY plane
        
        gate_yaw = np.arctan2(normal[1], normal[0]) 
        return gate_yaw
    
    def get_gate_index_from_position(self, pos):
        cx, cy = ARENA_CENTER[0], ARENA_CENTER[1]  # arena center

        dx = pos[0] - cx   # x: up
        dy = pos[1] - cy   # y: left

        # 0° = downward (-x), but now CLOCKWISE positive
        angle = np.atan2(-dy, -dx)

        deg = np.degrees(angle)
        if deg < 0:
            deg += 360

        # 12 sectors (30° each), centered
        sector = int((deg + 15) // 30) % 12 + 1

        mapping = {
            3: 0,
            5: 1,
            7: 2,
            9: 3,
            11: 4
        }

        # If it lands perfectly in a expected sector, return it
        if sector in mapping:
            return mapping[sector]

        # FALLBACK: Find the closest valid sector if it landed in a dead zone
        valid_sectors = list(mapping.keys())
        
        # Helper to calculate distance between sectors on a 12-hour clock face
        def sector_dist(s1, s2):
            diff = abs(s1 - s2)
            return min(diff, 12 - diff)

        # Snap to the sector with the minimum circular distance
        closest_sector = min(valid_sectors, key=lambda s: sector_dist(sector, s))
        
        # print(f"Warning: Gate detected in unmapped sector {sector}. Snapping to closest valid sector {closest_sector} (Index {mapping[closest_sector]}).")
        
        return mapping[closest_sector]
        
    def compute_approach_position(self, gate_center, gate_yaw):
        """
        Returns the 3-D point APPROACH_DIST metres behind the drone's facing
        direction (i.e. on the near side of the gate).
        """
        offset_x = -np.cos(gate_yaw) * APPROACH_DIST
        offset_y = -np.sin(gate_yaw) * APPROACH_DIST
        return np.array([gate_center[0] + offset_x,
                         gate_center[1] + offset_y,
                         gate_center[2]])
    
    def compute_fly_through_position(self, gate_center, gate_yaw):
        """
        Returns the 3-D point FLY_THROUGH_DIST metres beyond the gate center in the direction of the gate's facing.
        """
        offset_x = np.cos(gate_yaw) * FLY_THROUGH_DIST
        offset_y = np.sin(gate_yaw) * FLY_THROUGH_DIST
        return np.array([gate_center[0] + offset_x,
                         gate_center[1] + offset_y,
                         gate_center[2]])

    # --------------- Pause helpers ---------------
    def start_pause(self, duration, next_mode):
        self.pause_start_time = time.time()
        self.pause_duration = duration
        self.post_pause_mode = next_mode

    def is_pausing(self):
        if self.pause_start_time is None:
            return False
        if time.time() - self.pause_start_time < self.pause_duration:
            return True
        # Pause over — transition
        # print("Mode: Take Second Photo - Finished pause, taking second photo now")
        self.mode = self.post_pause_mode
        self.ready_to_take_second_photo = True
        self.pause_start_time = None
        self.pause_duration = None
        self.post_pause_mode = None
        return False
    
    # --------------- Geometry helpers ---------------
    def get_camera_position_in_world(self, sensor_data):
        """Get the world position of the camera given drone sensor data."""
        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Rotate the camera offset from body frame to world frame
        R_body_to_world = self.get_rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
        cam_offset_world = R_body_to_world @ CAM_OFFSET_BODY
        
        return drone_pos + cam_offset_world
    
    def get_rotation_matrix(self, roll, pitch, yaw):
        """ZYX Euler angles to rotation matrix (body to world)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr            ]
        ])

    def pixel_to_direction_vector(self, pixel_coords):
        """
        Convert a pixel (u, v) to a direction vector in camera frame.
        Moves origin from top-left to center, then uses focal length as z.
        pixel_coords: (u, v) in image coordinates
        """
        u, v = pixel_coords
        cx = CAM_WIDTH / 2   # principal point u0
        cy = CAM_HEIGHT  / 2   # principal point v0
        
        vx = u - cx  # x in camera frame
        vy = v - cy  # y in camera frame
        vz = F_PIXELS

        return np.array([vx, vy, vz], dtype=float)
    
    def compute_trajectory(self):
        key_points = [HOME_POSITION.copy()]
        
        gate_keys = sorted(self.gate_center_poses_dict.keys())
        for i, gate_idx in enumerate(gate_keys):
            center, gate_yaw = self.gate_center_poses_dict[gate_idx]

            if i == 0:
                prev_point = HOME_POSITION
            else:
                prev_center, _ = self.gate_center_poses_dict[gate_keys[i - 1]]
                prev_point = prev_center

            approach_vec = center[:2] - np.array(prev_point[:2])
            approach_dist = np.linalg.norm(approach_vec)
            if approach_dist > 1e-4:
                approach_dir = approach_vec / approach_dist
            else:
                approach_dir = np.array([np.cos(gate_yaw), np.sin(gate_yaw)])

            ALIGN_DIST = 0.4

            pre_gate = center.copy()
            pre_gate[0] -= approach_dir[0] * ALIGN_DIST
            pre_gate[1] -= approach_dir[1] * ALIGN_DIST

            post_gate = center.copy()
            post_gate[0] += approach_dir[0] * ALIGN_DIST
            post_gate[1] += approach_dir[1] * ALIGN_DIST

            key_points.append(pre_gate)
            key_points.append(center.copy())
            key_points.append(post_gate)

        key_points.append(HOME_POSITION.copy())
        self.plan_spline_trajectory(key_points)
        
        return self.trajectory_waypoints
    
    def plan_spline_trajectory(self, key_points):
        """
        Fits a smooth Cubic Spline using custom segment timing (turn penalties, 
        aggressive Z limits). Prevents loops by ensuring micro alignment-segments 
        are not subjected to artificial time floors.
        """
        n_segs = len(key_points) - 1
        points = np.array(key_points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # --- 1. Initial segment times based on 3D geometry + Custom Kinematic Limits ---
        seg_times = []
        for i in range(n_segs):
            p_curr = points[i]
            p_next = points[i+1]
            vec_fwd = p_next - p_curr
            
            dx, dy, dz = vec_fwd[0], vec_fwd[1], vec_fwd[2]
            dist_3d = np.linalg.norm(vec_fwd)
            
            # --- LOOP FIX: Bypass time inflation for tiny gate-alignment segments ---
            # Your pre/post gate points are ALIGN_DIST (0.4m) away. 
            # We strictly enforce proportional time here so the spline stays straight.
            if dist_3d < 1.0: 
                t_seg = dist_3d / TRAJ_SPEED
            else:
                # 1. Kinematically Decoupled Base Time for large segments
                dist_xy = np.sqrt(dx**2 + dy**2)
                t_xy = dist_xy / TRAJ_SPEED
                
                # 2. HYPER-AGGRESSIVE VERTICAL LIMITS
                if dz > 0:
                    t_z = dz / 1.5   # Punch the throttle!
                else:
                    t_z = abs(dz) / 1.2  # Free-fall into the low gates!
                    
                t_seg = max(t_xy, t_z)
                
                # 3. Add cornering time at the START of the segment
                if i > 0:
                    p_prev = points[i-1]
                    vec_in = p_curr - p_prev
                    t_seg += get_turn_penalty(vec_in, vec_fwd) * 0.5 
                    
                # 4. Add cornering time at the END of the segment
                if i < n_segs - 1:
                    p_next_next = points[i+2]
                    vec_out = p_next_next - p_next
                    t_seg += get_turn_penalty(vec_fwd, vec_out) * 0.5 

            seg_times.append(t_seg)

        # --- 2. Iteratively reduce times while checking dynamic limits ---
        for iteration in range(20):  
            # Convert segment durations to absolute time knots
            t_knots = np.insert(np.cumsum(seg_times), 0, 0.0)
            
            # Fit the Cubic Splines
            cs_x = CubicSpline(t_knots, x, bc_type='clamped')
            cs_y = CubicSpline(t_knots, y, bc_type='clamped')
            cs_z = CubicSpline(t_knots, z, bc_type='clamped')

            # Dense sampling to check limits 
            total_time = t_knots[-1]
            t_check = np.linspace(0, total_time, max(50, int(total_time / 0.05)))
            
            # Evaluate 1st (vel) and 2nd (acc) derivatives simultaneously 
            vx, vy, vz = cs_x(t_check, 1), cs_y(t_check, 1), cs_z(t_check, 1)
            ax, ay, az = cs_x(t_check, 2), cs_y(t_check, 2), cs_z(t_check, 2)
            
            vels = np.sqrt(vx**2 + vy**2 + vz**2)
            accs = np.sqrt(ax**2 + ay**2 + az**2)
            
            # Check maximums against your physical limits
            if np.max(vels) <= MAX_VELOCITY and np.max(accs) <= MAX_ACCELERATION:
                break
            else:
                # Limit exceeded: Scale all times up by 10% to slow down the drone
                seg_times = [t * 1.1 for t in seg_times]

        # --- 3. Final Spline Sampling for Execution ---
        total_time = t_knots[-1]
        t_samples = np.arange(0, total_time, TRAJ_DT)
        
        dense_x = cs_x(t_samples)
        dense_y = cs_y(t_samples)
        dense_z = cs_z(t_samples)

        self.trajectory_waypoints = np.vstack((dense_x, dense_y, dense_z)).T
        # print(f"Generated spline trajectory with {len(self.trajectory_waypoints)} waypoints. Total Time: {total_time:.2f}s")

        # =================== PLOT CODE (commented out) ===================
        # # --- Plot ---
        # fig = plt.figure(figsize=(12, 5))

        # # --- 1. 3D Path Plot ---
        # ax3d = fig.add_subplot(121, projection='3d')
        
        # # Plot the dense sampled path
        # wp = self.trajectory_waypoints
        # ax3d.plot(wp[:, 0], wp[:, 1], wp[:, 2], 'b-', linewidth=1, label='Spline Path')
        
        # # Plot the anchor key points
        # kp = np.array(key_points)
        # ax3d.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c='red', s=80, zorder=5)
        
        # for i, p in enumerate(key_points):
        #     label = 'Home' if (i == 0 or i == len(key_points)-1) else f'WP {i}'
        #     ax3d.text(p[0], p[1], p[2]+0.1, label, fontsize=8)
            
        # ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        # ax3d.set_title('Cubic Spline Trajectory')

        # # --- 2. Velocity Profile Plot ---
        # ax_v = fig.add_subplot(122)
        
        # # Evaluate the 1st derivative (1) of the splines to get velocity at every time sample
        # vx = cs_x(t_samples, 1)
        # vy = cs_y(t_samples, 1)
        # vz = cs_z(t_samples, 1)
        
        # # Calculate the magnitude of the velocity vector (speed)
        # speeds = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # ax_v.plot(t_samples, speeds, 'b-')
        
        # # Assuming MAX_VELOCITY is defined globally in your script
        # try:
        #     ax_v.axhline(MAX_VELOCITY, color='r', linestyle='--', label=f'v_max={MAX_VELOCITY}')
        # except NameError:
        #     pass # Skip if MAX_VELOCITY isn't imported here
            
        # ax_v.set_xlabel('Time (s)'); ax_v.set_ylabel('Speed (m/s)')
        # ax_v.set_title('Spline Velocity Profile'); ax_v.legend()

        # plt.tight_layout()
        # plt.savefig('trajectory.png')
        # # plt.show()
        
        # # Close the figure so it doesn't leak memory or lock up your background thread!
        # plt.close(fig)
        # =================== END OF PLOT CODE (commented out) ===================

    
    # --------------- Image processing helpers ---------------
    def locate_pink_area(self, image, lower_pink, upper_pink, min_area=100):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not contours:
            return None

        # Return a polygon for each contour
        gates = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            gates.append(polygon.reshape(-1, 2))

        return gates  # list of Nx2 arrays, one per gate

    def rgb_to_hsv_bounds(self, r, g, b, hue_tolerance=15, sat_min=40, val_min=80):
        """Helper to convert an RGB eyedropper reading into HSV bounds for inRange."""
        pixel = np.uint8([[[b, g, r]]])  # OpenCV is BGR
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        lower = np.array([max(h - hue_tolerance, 0),   max(s - 60, 0),   max(v - 60, 0)])
        upper = np.array([min(h + hue_tolerance, 180),  255,               255])
        return lower, upper

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def show_detection():
    pass
    # ============= GATE DETECTION VISUALIZATION (commented out) ==================
    # if _controller.gate_detection_img is not None:
    #     cv2.imshow("All Gate Detection", _controller.gate_detection_img)
    #     cv2.waitKey(1)
    # if _controller.target_gate_detection_img is not None:
    #     cv2.imshow("Target Gate Detection", _controller.target_gate_detection_img)
    #     cv2.waitKey(1)
    # =============== END OF GATE DETECTION VISUALIZATION (commented out) ==========



# ======================= Math functions ===========================
def solve_min_jerk_1d(waypoints, times):
    """
    Solve for minimum-jerk polynomial trajectory coefficients in 1D.
    Each segment i uses a 5th-degree polynomial: x(t) = sum(c[i,k] * t^k, k=0..5)
    where t is LOCAL time within the segment [0, T_i].

    Args:
        waypoints: list of m scalar positions [p0, p1, ..., p_{m-1}]
        times:     list of m-1 segment durations [T1, T2, ..., T_{m-1}]

    Returns:
        coeffs: np.array of shape (m-1, 6) — coefficients for each segment
    """
    m = len(waypoints)
    n_segs = m - 1
    n_coeffs = 6 * n_segs  # 6 per segment (degree 5 polynomial)

    A = np.zeros((n_coeffs, n_coeffs))
    b = np.zeros(n_coeffs)

    row = 0

    # ---------- Initial boundary conditions (segment 0, local t=0) ----------
    # Position
    A[row, 0] = 1
    b[row] = waypoints[0]
    row += 1
    # Velocity = 0
    A[row, 1] = 1
    b[row] = 0
    row += 1
    # Acceleration = 0
    A[row, 2] = 2
    b[row] = 0
    row += 1

    # ---------- Final boundary conditions (last segment, local t=T) ----------
    seg = n_segs - 1
    T = times[seg]
    base = seg * 6
    # Position
    A[row, base:base+6] = [1, T, T**2, T**3, T**4, T**5]
    b[row] = waypoints[-1]
    row += 1
    # Velocity = 0
    A[row, base:base+6] = [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4]
    b[row] = 0
    row += 1
    # Acceleration = 0
    A[row, base:base+6] = [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    b[row] = 0
    row += 1

    # ---------- Intermediate waypoint constraints ----------
    for seg in range(n_segs - 1):
        T = times[seg]
        base_curr = seg * 6
        base_next = (seg + 1) * 6

        # End position of current segment = waypoint
        A[row, base_curr:base_curr+6] = [1, T, T**2, T**3, T**4, T**5]
        b[row] = waypoints[seg + 1]
        row += 1

        # Start position of next segment = same waypoint (continuity)
        A[row, base_next] = 1
        b[row] = waypoints[seg + 1]
        row += 1

        # Continuity of velocity
        A[row, base_curr:base_curr+6] = [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4]
        A[row, base_next + 1] = -1
        b[row] = 0
        row += 1

        # Continuity of acceleration
        A[row, base_curr:base_curr+6] = [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        A[row, base_next + 2] = -2
        b[row] = 0
        row += 1

        # Continuity of jerk
        A[row, base_curr:base_curr+6] = [0, 0, 0, 6, 24*T, 60*T**2]
        A[row, base_next + 3] = -6
        b[row] = 0
        row += 1

        # Continuity of snap (4th derivative)
        A[row, base_curr:base_curr+6] = [0, 0, 0, 0, 24, 120*T]
        A[row, base_next + 4] = -24
        b[row] = 0
        row += 1

    coeffs_flat = np.linalg.solve(A, b)
    return coeffs_flat.reshape(n_segs, 6)


def eval_poly_1d(coeffs_seg, t_local):
    """Evaluate position of a single polynomial segment at local time t."""
    t = t_local
    return sum(coeffs_seg[k] * t**k for k in range(6))


def eval_poly_vel_1d(coeffs_seg, t_local):
    """Evaluate velocity (1st derivative) of a polynomial segment at local time t."""
    t = t_local
    return coeffs_seg[1] + 2*coeffs_seg[2]*t + 3*coeffs_seg[3]*t**2 + \
           4*coeffs_seg[4]*t**3 + 5*coeffs_seg[5]*t**4


def eval_poly_acc_1d(coeffs_seg, t_local):
    """Evaluate acceleration (2nd derivative) of a polynomial segment at local time t."""
    t = t_local
    return 2*coeffs_seg[2] + 6*coeffs_seg[3]*t + 12*coeffs_seg[4]*t**2 + \
           20*coeffs_seg[5]*t**3

def get_turn_penalty(vec_in, vec_out):
    """
    Calculates the 3D angle between two vectors and returns a time penalty in seconds.
    """
    norm_in = np.linalg.norm(vec_in)
    norm_out = np.linalg.norm(vec_out)
    
    if norm_in < 1e-4 or norm_out < 1e-4:
        return 0.0
        
    cos_theta = np.clip(np.dot(vec_in, vec_out) / (norm_in * norm_out), -1.0, 1.0)
    theta = np.arccos(cos_theta) 
    MAX_TURN_PENALTY = MAX_TURN_PENALTY = 0.2
    
    return (theta / np.pi) * MAX_TURN_PENALTY