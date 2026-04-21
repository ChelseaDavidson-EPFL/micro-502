import numpy as np
import time
import cv2

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
    SEARCH_GATE = 4
    APPROACH_GATE = 5
    TAKE_SECOND_PHOTO = 6
    FLY_THROUGH_GATE = 7

# Pink gates
gates_r_value = 211
gates_g_value = 144
gates_b_value = 222
GATE_HEIGHT = 0.4 # In meters 

# Rotation amounts
search_gate_rotation = 0.15
search_gate_translation = 0.1

# Approach distance
APPROACH_DIST = 0.8  # metres in front of gate to take measurement

# Fly through distance - how far forward to go after the gate center when flying through
FLY_THROUGH_DIST = 0.1

# Thresholds
eps = 0.02 # Used to check if yaw is correct - Approx 1 degree
pos_eps = 0.05 # Allow 5cm variation in approaching gate
edge_threshold = 10  # Used to check if the detected gate is at the edge (pixels tolerance from edge)

# Pause durations
PAUSE_AT_MEASUREMENT_POS = 3.0  # seconds — wait before taking 2nd photo
PAUSE_AT_GATE_CENTER     = 1.5  # seconds — wait after reaching gate center

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

class MyAssignment:
    def __init__(self, ):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.mode = Mode.TAKE_OFF
        self.has_taken_off = False
        self.gate_corners = [] # Store gates as tuples of tuples representing coords of the corners ((x, y, z), ... x 4)
        self.gate_center_poses = [] # Store just the centers of the gates and their orientations for easy access when flying through
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

    # Pause helpers 
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
        print("Mode: Take Second Photo - Finished pause, taking second photo now")
        self.mode = self.post_pause_mode
        self.ready_to_take_second_photo = True
        self.pause_start_time = None
        self.pause_duration = None
        self.post_pause_mode = None
        return False


    def compute_command(self, sensor_data, camera_data, dt):
        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Default control command - TODO: remove this later 
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
        
        # Take off command
        if (self.mode == Mode.TAKE_OFF):
            if sensor_data['z_global'] < 0.49 and not self.has_taken_off:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
                return control_command
            if not self.has_taken_off and sensor_data['z_global'] > 0.49:
                self.has_taken_off = True
                self.mode = Mode.SEARCH_GATE
                print("Mode: Search Gate")
        # Search for gate command
        elif (self.mode == Mode.SEARCH_GATE):
            control_command = self.get_search_gate_command(camera_data, sensor_data)  
        elif (self.mode == Mode.APPROACH_GATE):
            control_command = self.get_approach_gate_command(camera_data, sensor_data) 
        elif (self.mode == Mode.FLY_THROUGH_GATE):
            control_command = self.get_fly_through_gate_command(sensor_data)
        elif (self.mode == Mode.TAKE_SECOND_PHOTO):
            control_command = self.get_capture_second_photo_command(camera_data, sensor_data)

        # ---- YOUR CODE HERE ----
        # self.get_move_to_gate_command(camera_data)
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


    def get_target_gate(self, camera_data):
        gates = self.locate_gates(camera_data)
        
        if gates is None:
            #Turn to left
            return None
        # Check if more than 1 gate 
        if len(gates) == 1: # Found a gate so now try and take a 2nd photo 
            target_gate = gates[0] 
        else:
            # Multiple gates — pick rightmost by center x - TODO: maybe this should be based on world coords not camera
            target_gate = max(gates, key=lambda g: g[:, 0].mean())

        return target_gate

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

        if touching_left and not touching_right:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation]
        elif touching_right and not touching_left:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] - search_gate_rotation]
        elif touching_top and not touching_bottom:
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
        else:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]



    def get_search_gate_command(self, camera_data, sensor_data):
        target_gate = self.get_target_gate(camera_data)
        if target_gate is None:
            # No gate in view, keep rotating to search
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation]
        
        if self.is_target_gate_not_fully_in_FOV(camera_data, target_gate):
            print("Search Gate: Target gate not fully in view, adjusting position")
            return self.adjust_position_for_better_FOV(camera_data, sensor_data, target_gate)
        
        self.target_gate_detection_img = camera_data.copy()
        self.target_gate_detection_img = cv2.polylines(self.target_gate_detection_img, [target_gate], isClosed=True, color=(0, 0, 255), thickness=2)

        gate_corners = self.estimate_gate_position(target_gate, sensor_data)
        if gate_corners is None:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        center = gate_corners.mean(axis=0)
        gate_yaw = self.estimate_gate_orientation(gate_corners, sensor_data)

        approach_pos = self.compute_approach_position(center, gate_yaw)

        self.measurement_target_pos = approach_pos
        self.measurement_target_yaw = gate_yaw

        self.mode = Mode.APPROACH_GATE
        print(f"Mode: Approach Gate — target={approach_pos}, gate_yaw={np.degrees(gate_yaw):.1f}°")
        return [approach_pos[0], approach_pos[1], approach_pos[2], gate_yaw]

    def get_approach_gate_command(self, camera_data, sensor_data):
        if self.measurement_target_pos is None or self.measurement_target_yaw is None:
            self.mode = Mode.SEARCH_GATE
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Check how far we are from our 0.8m measurement spot
        dist_to_target = np.linalg.norm(self.measurement_target_pos - drone_pos)

        # Check yaw error (wrapped to [-π, π])
        yaw_error = abs((sensor_data['yaw'] - self.measurement_target_yaw + np.pi) % (2 * np.pi) - np.pi)
        facing_gate = yaw_error < eps
        
        # If we are within 10cm of the measurement point, take the new photo
        if dist_to_target < pos_eps and facing_gate:
            self.mode = Mode.TAKE_SECOND_PHOTO
            print("Mode: Take Second Photo - Pausing to take second photo at measurement position")
            self.start_pause(PAUSE_AT_MEASUREMENT_POS, Mode.TAKE_SECOND_PHOTO) 
        elif dist_to_target < pos_eps and not facing_gate:
            print(f"Approach: at position but yaw not settled yet (error={np.degrees(yaw_error):.1f}°), waiting")
            
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
        target_gate = self.get_target_gate(camera_data)
        if (target_gate is None):
            # If we lost the gate, go back to searching
            self.mode = Mode.SEARCH_GATE
            self.ready_to_take_second_photo = False
            print("Mode: Search Gate (gate lost at measurement pos)")
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation]
        
        if self.is_target_gate_not_fully_in_FOV(camera_data, target_gate):
            print("Second photo: Target gate not fully in view, adjusting position")
            return self.adjust_position_for_better_FOV(camera_data, sensor_data, target_gate)
        
        gate_corners = self.estimate_gate_position(target_gate, sensor_data)
        
        if gate_corners is not None:
            self.gate_corners.append(gate_corners)
            
            center = gate_corners.mean(axis=0)
            gate_yaw = self.estimate_gate_orientation(gate_corners, sensor_data)
            self.gate_center_poses.append((center, gate_yaw))
            print(f"Final High-Accuracy Gate Center: {center}")
            print(f"Final High-Accuracy Gate Yaw: {np.degrees(gate_yaw):.1f}°")

            self.mode = Mode.FLY_THROUGH_GATE
            print("Mode: Fly Through Gate")
            return [center[0], center[1], center[2], gate_yaw] # Don't set it as 0.1m beyond yet - we'll do that in the fly through state to ensure we are stable at the gate center before flying through        

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
            self.mode = Mode.SEARCH_GATE
            print("Mode: Search Gate")
        
        return [target_x, target_y, target_z, gate_yaw]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
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


    # def get_move_to_gate_command(self, camera_data): #TODO - this will be different for later laps
    #     #TODO - call locate gates, turn to left if there are none, choose rightmost if multiple, convert to world frame
    #     gates = self.locate_gates(camera_data)
        
    #     # Check to see which is next and convert to world
    #     if gates is None:
    #         #TODO - turn to left
    #         return
    #     #Check if more than 1 gate 
    #     if len(gates) == 1:
    #         target_gate = gates[0]
    #     else:
    #         # Multiple gates — pick rightmost by center x - TODO: maybe this should be based on world coords not camera
    #         target_gate = max(gates, key=lambda g: g[:, 0].mean())
        
    #     # Store the target gate to show later 
    #     self.target_gate_detection_img = camera_data.copy()
    #     self.target_gate_detection_img = cv2.polylines(self.target_gate_detection_img, [target_gate], isClosed=True, color=(0, 0, 255), thickness=2)
        
    #     # TODO - convert target_gate corners to world frame
    #     return

    def convert_pixel_to_world(self, pixel_coords):
        #TODO
        return

    def locate_gates(self, camera_data):
        # TODO - call get image and locate corners in image
        lower, upper = self.rgb_to_hsv_bounds(r=gates_r_value, g=gates_g_value, b=gates_b_value)
        gates = self.locate_pink_area(camera_data, lower, upper)

        if gates is None:
            return None

        # Filter to only gates with exactly 4 corners
        valid_gates = [g for g in gates if len(g) == 4]

        if not valid_gates:
            return None

        for points in valid_gates:
            self.gate_detection_img = camera_data.copy()
            self.gate_detection_img = cv2.polylines(self.gate_detection_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return valid_gates # list of polygons, or None
    
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

    def rgb_to_hsv_bounds(self, r, g, b, hue_tolerance=10, sat_min=40, val_min=80):
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
    if _controller.gate_detection_img is not None:
        cv2.imshow("All Gate Detection", _controller.gate_detection_img)
        cv2.waitKey(1)
    if _controller.target_gate_detection_img is not None:
        cv2.imshow("Target Gate Detection", _controller.target_gate_detection_img)
        cv2.waitKey(1)

