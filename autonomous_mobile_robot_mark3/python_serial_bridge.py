#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
import tf_transformations
import tf2_ros
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

        # ROS publishers/subscribers
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odom', 10)  # Changed to /wheel/odom
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10) 
        self.latest_imu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Serial
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.ser.flushInput()
        self.ser.flushOutput()
        self.get_logger().info(f"Opened serial port {port} @ {baud}")

        # Robot params
        self.ticks_per_rev = float(ticks_per_rev)
        self.r = float(wheel_radius)
        self.L = float(wheel_base) / 2.0
        self.W = float(track_width) / 2.0

        # Encoder and timing
        self.prev_ticks = [0, 0, 0, 0]  # FL, FR, RR, RL
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vth = 0.0
        self.prev_time = time.time()

        # Latest encoder values (updated by thread)
        self.latest_ticks = [0, 0, 0, 0]
        self.lock = threading.Lock()

        # Start background serial reading thread
        self.thread = threading.Thread(target=self.serial_read_thread, daemon=True)
        self.thread.start()

        # Odometry timer
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.timer_callback)

    def cmd_vel_callback(self, msg: Twist):
        """Send cmd_vel to Arduino immediately"""
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z
        command = f"{vx:.3f},{vy:.3f},{omega:.3f}\n"
        try:
            self.ser.write(command.encode('utf-8'))
            self.get_logger().info(f"Sent cmd_vel: vx={vx:.3f}, vy={vy:.3f}, omega={omega:.3f}")
        except Exception as e:
            self.get_logger().error(f"Serial write error: {e}")

    def serial_read_thread(self):
        """Continuously read serial lines and update latest encoder values"""
        while rclpy.ok():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('S'):
                    parts = line.split(',')
                    if len(parts) >= 11:
                        try:
                            e1, e2, e3, e4 = map(float, parts[7:11])  # FL, FR, RR, RL
                            gx, gy, gz, ax, ay, az = map(float, parts[1:7])
                            with self.lock:
                                self.latest_ticks = [e1, e2, e3, e4]
                                self.latest_imu = [gx, gy, gz, ax, ay, az]
                        except ValueError:
                            self.get_logger().warn(f"Invalid encoder data: {line}")
            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}")
                time.sleep(0.5)
            except ValueError:
                self.get_logger().warn(f"Invalid data: {line}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")

    def timer_callback(self):
        """Compute odometry from latest encoder ticks and publish"""
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0.0:
            return
        self.prev_time = now

        # Copy latest ticks safely
        with self.lock:
            ticks = self.latest_ticks.copy()
            gx, gy, gz, ax, ay, az = self.latest_imu

        delta_ticks = [ticks[i] - self.prev_ticks[i] for i in range(4)]
        self.prev_ticks = ticks

        # Compute wheel velocities
        omega_w = [(delta / self.ticks_per_rev) * (2.0 * math.pi) / dt for delta in delta_ticks]
        v_w = [w * self.r for w in omega_w]  # linear wheel velocities
        v_FL, v_FR, v_RR, v_RL = v_w

        # Mecanum inverse kinematics
        self.vx = (v_FL + v_FR + v_RR + v_RL) / 4.0
        self.vy = (-v_FL + v_FR + v_RL - v_RR) / 4.0
        self.vth = (-v_FL + v_FR - v_RL + v_RR) / (4.0 * (self.L + self.W))

        if abs(self.vx) > 0.01 or abs(self.vy) > 0.01 or abs(self.vth) > 0.01: 
            self.get_logger().info(f"Wheel odom velocities: vx={self.vx:.3f}, vy={self.vy:.3f}, omega={self.vth:.3f}")

        # Integrate pose 
        dx = (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        dy = (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        d_yaw = self.vth * dt
        self.x += dx
        self.y += dy
        self.yaw += d_yaw

        # Publish wheel odometry
        current_ros_time = self.get_clock().now().to_msg()
        odom_msg = Odometry()
        odom_msg.header.stamp = current_ros_time
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        odom_msg.twist.twist.linear.x = self.vx
        odom_msg.twist.twist.linear.y = self.vy
        odom_msg.twist.twist.angular.z = self.vth

        # ===== IMPROVED COVARIANCE FOR MECANUM WHEELS =====
        # Higher uncertainty in lateral (y) and rotational motion due to slip
        # Order: x, y, z, roll, pitch, yaw
        odom_msg.pose.covariance = [
            0.5,  0.0,  0.0,  0.0,  0.0,  0.0,    # x - moderate uncertainty
            0.0,  2.0,  0.0,  0.0,  0.0,  0.0,    # y - HIGH uncertainty (lateral slip)
            0.0,  0.0,  1e6,  0.0,  0.0,  0.0,    # z - not used
            0.0,  0.0,  0.0,  1e6,  0.0,  0.0,    # roll - not used
            0.0,  0.0,  0.0,  0.0,  1e6,  0.0,    # pitch - not used
            0.0,  0.0,  0.0,  0.0,  0.0,  1e6     # yaw - DON'T trust from wheels!
        ]

        # Twist covariance - velocities also unreliable
        odom_msg.twist.covariance = [
            1.0,  0.0,  0.0,  0.0,  0.0,  0.0,    # vx - moderate uncertainty
            0.0,  5.0,  0.0,  0.0,  0.0,  0.0,    # vy - VERY HIGH uncertainty (slip!)
            0.0,  0.0,  1e6,  0.0,  0.0,  0.0,    # vz - not used
            0.0,  0.0,  0.0,  1e6,  0.0,  0.0,    # vroll - not used
            0.0,  0.0,  0.0,  0.0,  1e6,  0.0,    # vpitch - not used
            0.0,  0.0,  0.0,  0.0,  0.0,  1e6     # vyaw - DON'T trust from wheels!
        ]

        self.odom_pub.publish(odom_msg)

        # ===== PUBLISH IMU WITH PROPER COVARIANCES =====
        imu_msg = Imu()
        imu_msg.header.stamp = current_ros_time
        imu_msg.header.frame_id = "imu_link"
        
        # Angular velocity (gyroscope) - THIS IS ACCURATE!
        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        
        # Linear acceleration
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        # Orientation covariance - set to -1 to indicate no orientation data
        imu_msg.orientation_covariance = [-1.0] + [0.0]*8

        # Angular velocity covariance - LOW values = HIGH TRUST
        # Gyros are typically very accurate!
        imu_msg.angular_velocity_covariance = [
            1e6, 0.0, 0.0,   # roll rate - not used in 2D
            0.0, 1e6, 0.0,   # pitch rate - not used in 2D
            0.0, 0.0, 0.01   # yaw rate - TRUST THIS! (0.01 = very accurate)
        ]
        
        # Linear acceleration covariance
        imu_msg.linear_acceleration_covariance = [
            0.5, 0.0, 0.0,   # ax - moderate trust
            0.0, 0.5, 0.0,   # ay - moderate trust
            0.0, 0.0, 1e6    # az - not used in 2D
        ]

        self.imu_pub.publish(imu_msg)

        # Publish TF (this will be overridden by EKF output)
        transform = TransformStamped()
        transform.header.stamp = current_ros_time
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    node = OdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
