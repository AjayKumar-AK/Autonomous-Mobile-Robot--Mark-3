import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class ImpactDetector(Node):
    def __init__(self):
        super().__init__('impact_detector')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.listener_callback, 10)
        self.prev_ranges = None
        self.wall_baseline = 2.0  # Adjust to your wall distance (e.g., from empty scans)
        self.impact_threshold = 0.2  # Min range drop for detection
        self.publisher = self.create_publisher(LaserScan, '/impacts', 10)  # Optional: publish detections

    def listener_callback(self, msg):
        current_ranges = np.array(msg.ranges)
        current_ranges[np.isinf(current_ranges)] = msg.range_max
        if self.prev_ranges is not None:
            deltas = self.prev_ranges - current_ranges
            impacts = np.where((deltas > self.impact_threshold) & (current_ranges < self.wall_baseline))[0]
            if len(impacts) > 0:
                angles = np.array([msg.angle_min + i * msg.angle_increment for i in impacts])
                self.get_logger().info(f'Impact detected at angles: {angles}')
                # Here: Trigger display update (see step 3)
        self.prev_ranges = current_ranges

def main(args=None):
    rclpy.init(args=args)
    node = ImpactDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
'''#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf_transformations
import serial
import threading
import time
import math

class OdomBridge(Node):
    def __init__(self,
                 port='/dev/ttyUSB0',
                 baud=115200,
                 ticks_per_rev=490.0,
                 wheel_radius=0.047,
                 wheel_base=0.164,
                 track_width=0.22,
                 publish_rate_hz=50.0):
        super().__init__('odom_bridge')

        # ===== ROS PUBLISHERS =====
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # ===== SERIAL =====
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.ser.flushInput()
        self.ser.flushOutput()
        self.get_logger().info(f"Opened serial port {port} @ {baud}")

        # ===== ROBOT PARAMS =====
        self.ticks_per_rev = float(ticks_per_rev)
        self.r = float(wheel_radius)
        self.L = float(wheel_base) / 2.0
        self.W = float(track_width) / 2.0

        # ===== STATE =====
        self.prev_ticks = [0, 0, 0, 0]
        self.latest_ticks = [0, 0, 0, 0]
        self.latest_imu = [0.0]*6

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vth = 0.0
        self.prev_time = time.time()

        self.lock = threading.Lock()

        # ===== THREADS =====
        threading.Thread(target=self.serial_read_thread, daemon=True).start()
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.timer_callback)

    def cmd_vel_callback(self, msg: Twist):
        command = f"{msg.linear.x:.3f},{msg.linear.y:.3f},{msg.angular.z:.3f}\n"
        try:
            self.ser.write(command.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Serial write error: {e}")

    def serial_read_thread(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('S'):
                    parts = line.split(',')
                    if len(parts) >= 11:
                        with self.lock:
                            self.latest_imu = list(map(float, parts[1:7]))
                            self.latest_ticks = list(map(float, parts[7:11]))
            except Exception:
                pass

    def timer_callback(self):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0.0:
            return
        self.prev_time = now

        with self.lock:
            ticks = self.latest_ticks.copy()
            gx, gy, gz, ax, ay, az = self.latest_imu

        delta_ticks = [ticks[i] - self.prev_ticks[i] for i in range(4)]
        self.prev_ticks = ticks

        omega_w = [(d / self.ticks_per_rev) * (2.0 * math.pi) / dt for d in delta_ticks]
        v_w = [w * self.r for w in omega_w]
        v_FL, v_FR, v_RR, v_RL = v_w

        self.vx = (v_FL + v_FR + v_RR + v_RL) / 4.0
        self.vy = (-v_FL + v_FR + v_RL - v_RR) / 4.0
        self.vth = (-v_FL + v_FR - v_RL + v_RR) / (4.0 * (self.L + self.W))

        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        self.yaw += self.vth * dt

        # ===== ODOM MESSAGE =====
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.angular.z = self.vth

        self.odom_pub.publish(odom)

        # ===== IMU MESSAGE =====
        imu = Imu()
        imu.header.stamp = odom.header.stamp
        imu.header.frame_id = "imu_link"
        imu.angular_velocity.z = gz
        imu.linear_acceleration.x = ax
        imu.linear_acceleration.y = ay
        imu.linear_acceleration.z = az
        imu.orientation_covariance[0] = -1.0

        self.imu_pub.publish(imu)

def main():
    rclpy.init()
    node = OdomBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()'''

