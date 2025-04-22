#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import mediapipe as mp

class GestureDroneController(Node):
    def __init__(self):
        super().__init__('gesture_drone_controller')
        
        # Create publisher for drone velocity commands
        self.velocity_publisher = self.create_publisher(
            Twist,
            '/ImprovedDrone/cmd_vel',
            10
        )
        
        # Initialize MediaPipe Hands with better parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Try to open camera (try multiple devices)
        self.cap = None
        for i in range(2):  # Try both video0 and video1
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                self.get_logger().info(f'Successfully opened camera device /dev/video{i}')
                break
            else:
                self.cap.release()
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error('Failed to open any camera device')
            raise RuntimeError('Failed to open camera')
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.get_logger().info('Camera initialized successfully')
        
        # Create timer for camera callback
        self.timer = self.create_timer(0.2, self.camera_callback)
        
        # Initialize camera window
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', 640, 480)
        self.get_logger().info('OpenCV window created')
        
        self.get_logger().info('Gesture Drone Controller has been initialized')
        
        # Initialize current gesture
        self.current_gesture = None
        self.gesture_confidence = 0
        self.gesture_threshold = 5
    
    def is_finger_extended(self, tip, pip, mcp):
        # A finger is extended if its tip is above its PIP joint
        return tip.y < pip.y
    
    def get_gesture(self, landmarks):
        if not landmarks:
            return None
            
        # Get key points for thumb and index
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Get key points for other fingers
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_mcp = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_mcp = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate distances between fingers
        thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - 
                                        np.array([index_tip.x, index_tip.y]))
        
        # Check if fingers are extended using PIP joints
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        index_extended = index_tip.y < index_pip.y
        thumb_extended = thumb_tip.y < thumb_ip.y
        
        # Calculate finger angles relative to wrist
        thumb_angle = np.arctan2(thumb_tip.y - wrist.y, thumb_tip.x - wrist.x)
        index_angle = np.arctan2(index_tip.y - wrist.y, index_tip.x - wrist.x)
        
        # Check OK sign first (most specific)
        if (thumb_index_dist < 0.06 and  # Thumb and index very close
            not thumb_extended and       # Thumb not extended
            not index_extended and       # Index not extended
            middle_extended and          # Middle finger extended
            ring_extended and            # Ring finger extended
            pinky_extended):             # Pinky extended
            return 'OK'
        
        # Check thumbs down (thumb extended and pointing down, other fingers not extended)
        elif (thumb_tip.y > wrist.y + 0.1 and  # Thumb clearly below wrist
              thumb_angle > 0.7 and            # Thumb pointing down
              not index_extended and
              not middle_extended and
              not ring_extended and
              not pinky_extended):
            return 'Thumbs Down'
        
        # Check closed fist (all fingers not extended and thumb not touching index)
        elif (not thumb_extended and
              not index_extended and
              not middle_extended and
              not ring_extended and
              not pinky_extended and
              thumb_index_dist > 0.1):  # Thumb not touching index
            return 'Close'
        
        # Check open palm (all fingers extended)
        elif (thumb_extended and
              index_extended and
              middle_extended and
              ring_extended and
              pinky_extended):
            return 'Open'
        
        # Check thumbs up (only thumb extended, pointing up)
        elif (thumb_extended and
              not index_extended and
              not middle_extended and
              not ring_extended and
              not pinky_extended and
              thumb_angle < -0.7):
            return 'Thumbs Up'
        
        # Check pointing (only index extended)
        elif (index_extended and
              not middle_extended and
              not ring_extended and
              not pinky_extended):
            if index_angle > 0.3:
                return 'Point Right'
            elif index_angle < -0.3:
                return 'Point Left'
        
        # Default to closed hand if no other gesture detected
        return 'Close'
    
    def camera_callback(self):
        try:
            # Get frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('Failed to get frame from webcam')
                return
            
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Get gesture
            gesture = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = self.get_gesture(hand_landmarks)
                    if gesture:
                        # Gesture confirmation logic
                        if gesture == self.current_gesture:
                            self.gesture_confidence = min(self.gesture_confidence + 1, self.gesture_threshold)
                        else:
                            self.gesture_confidence = max(self.gesture_confidence - 1, 0)
                            if self.gesture_confidence == 0:
                                self.current_gesture = gesture
                        
                        # Display gesture on frame with larger text
                        confidence_color = (0, 255, 0) if self.gesture_confidence >= self.gesture_threshold else (0, 165, 255)
                        cv2.putText(frame, f"Gesture: {gesture}", (20, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, confidence_color, 2)
                        
                        # Draw hand landmarks with confidence-based color
                        self.mp_drawing = mp.solutions.drawing_utils
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=confidence_color, thickness=2, circle_radius=2))
            
            # Display the image
            cv2.imshow('Camera Feed', frame)
            
            # Publish velocity command based on gesture (only when confident)
            if self.current_gesture and self.gesture_confidence >= self.gesture_threshold:
                cmd_vel = Twist()
                
                # Set linear and angular velocities based on gesture
                if self.current_gesture == 'Thumbs Up':
                    cmd_vel.linear.z = 1.0  # Move up
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                elif self.current_gesture == 'Thumbs Down':
                    cmd_vel.linear.z = -1.0  # Move down
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                elif self.current_gesture == 'Point Right':
                    cmd_vel.linear.x = 1.0  # Move forward
                    cmd_vel.linear.z = 0.0
                    cmd_vel.angular.z = 0.0
                elif self.current_gesture == 'Point Left':
                    cmd_vel.linear.x = -1.0  # Move backward
                    cmd_vel.linear.z = 0.0
                    cmd_vel.angular.z = 0.0
                elif self.current_gesture == 'Open':
                    cmd_vel.angular.z = 1.0  # Rotate right
                    cmd_vel.linear.x = 0.0
                    cmd_vel.linear.z = 0.0
                elif self.current_gesture == 'Close':
                    cmd_vel.angular.z = -1.0  # Rotate left
                    cmd_vel.linear.x = 0.0
                    cmd_vel.linear.z = 0.0
                elif self.current_gesture == 'OK':
                    # Stop all movement
                    cmd_vel.linear.x = 0.0
                    cmd_vel.linear.z = 0.0
                    cmd_vel.angular.z = 0.0
                
                # Publish the velocity command
                self.velocity_publisher.publish(cmd_vel)
                self.get_logger().info(f'Publishing velocity command for gesture: {self.current_gesture}')
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                self.get_logger().info('ESC pressed, shutting down')
                self.destroy_node()
                rclpy.shutdown()
                return
            
        except Exception as e:
            self.get_logger().error(f'Error in camera callback: {str(e)}')

    def __del__(self):
        # Clean up OpenCV windows and release webcam when the node is destroyed
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        cv2.destroyAllWindows()

def main():
    # Initialize ROS2 and create node
    rclpy.init()
    
    # Create and run the node
    node = GestureDroneController()
    
    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 