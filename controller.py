from src.PID_controller import PIDController
import numpy as np
from scipy.linalg import solve_continuous_are


"""
# 初始化PID控制器（参数需调试）
pid_x = PIDController(Kp=1.0, Ki=0.1, Kd=0.2, Ki_sat=[10, 10, 10])
pid_y = PIDController(Kp=1.0, Ki=0.1, Kd=0.2, Ki_sat=[10, 10, 10])
pid_z = PIDController(Kp=1.0, Ki=0.1, Kd=0.2, Ki_sat=[10, 10, 10])
pid_yaw = PIDController(Kp=0.5, Ki=0.05, Kd=0.1, Ki_sat=[1, 1, 1])

# Implement a controller
def controller(state, target_pos, dt):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    
    # get current state and target
    x, y, z, roll, pitch, yaw = state
    x_target, y_target, z_target, yaw_target = target_pos

    
    # calculate error
    error_x = x_target - x
    error_y = y_target - y
    error_z = z_target - z
    error_yaw = yaw_target - yaw


    # PID control
    vel_x = pid_x.control_update(np.array([error_x, 0, 0]), dt)[0]
    vel_y = pid_y.control_update(np.array([error_y, 0, 0]), dt)[0]
    vel_z = pid_z.control_update(np.array([error_z, 0, 0]), dt)[0]
    yaw_rate = pid_yaw.control_update(np.array([error_yaw, 0, 0]), dt)[0]


    # clip output
    vel_x = np.clip(vel_x, -1, 1)
    vel_y = np.clip(vel_y, -1, 1)
    vel_z = np.clip(vel_z, -1, 1)
    yaw_rate = np.clip(yaw_rate, -1.74533, 1.74533)

    return (vel_x, vel_y, vel_z, yaw_rate)

    #output = (1, 1, 1, 0.001)
    #return output

"""

class CascadeController:
    def __init__(self):

        # 外环PID：控制位置（输出为期望姿态角）
        # Outer loop PID: Position control (outputs desired attitude)
        self.pid_x = PIDController(Kp=1.0, Ki=0.01, Kd=0.1, Ki_sat=[0.5, 0.5, 0.5])
        self.pid_y = PIDController(Kp=1.0, Ki=0.01, Kd=0.1, Ki_sat=[0.5, 0.5, 0.5])
        self.pid_z = PIDController(Kp=2.0, Ki=0.05, Kd=0.2, Ki_sat=[1, 1, 1])
        
        # 内环PID：控制姿态（由 tello_controller.py 实现）
        # Inner loop PID: Attitude control (implemented in tello_controller.py)

    def controller(self, state, target_pos, dt):
        # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
        # target_pos format: (x (m), y (m), z (m), yaw (radians))
        # dt: time step (s)
        # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))

        # current state
        x, y, z, roll, pitch, yaw = state
        x_target, y_target, z_target, yaw_target = target_pos

        # Outer loop PID: Position control
        error_x = x_target - x
        error_y = y_target - y
        error_z = z_target - z

        # 计算期望速度（外环输出）
        # Compute desired velocity (outer loop output)
        vel_x = self.pid_x.control_update(np.array([error_x, 0, 0]), dt)[0]
        vel_y = self.pid_y.control_update(np.array([error_y, 0, 0]), dt)[0]
        vel_z = self.pid_z.control_update(np.array([error_z, 0, 0]), dt)[0]

        # 将水平速度转换为期望姿态角
        # Convert horizontal velocity to desired attitude (key step)
        roll_desired = -vel_y  # Roll angle (sign adjusted for coordinate system) 根据坐标系方向调整符号
        pitch_desired = vel_x  # Pitch angle
        thrust = vel_z + 0.6  # Base thrust (adjust based on drone mass) 基础推力（需根据无人机质量调整）

        #  Clamp outputs to safe ranges
        roll_desired = np.clip(roll_desired, -0.5, 0.5)  # Limit roll to ±0.5 rad ±30度以内
        pitch_desired = np.clip(pitch_desired, -0.5, 0.5)
        thrust = np.clip(thrust, 0.3, 0.8)

        #Return velocity commands (inner loop handled by tello_controller.py)
        return (vel_x, vel_y, vel_z, 0)  # Yaw rate temporarily set to 0, yaw_rate暂设为0

# instance
cascade_controller = CascadeController()




###########################################################################################################
#        LQR
############################################################################################################

class LQRController:
    def __init__(self):
        """初始化LQR控制器 / Initialize LQR controller"""
        # 定义状态空间模型 / Define state-space model (simplified hover point linearization)
        self.n_states = 12  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.n_controls = 4  # [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
        
        self.A = np.zeros((12, 12))
        self.A[0:3, 3:6] = np.eye(3)  # Position -> velocity
        self.A[3:5, 6:8] = np.array([[0, 9.81], [-9.81, 0]])  # Gravity effect on pitch/roll
        self.A[6:9, 9:12] = np.eye(3)  # Attitude -> angular rates
    
        # 控制矩阵B (12x4)
        self.B = np.zeros((12, 4))
        self.B[3:6, 0:3] = np.eye(3)  # vx,vy,vz -> acceleration
        self.B[9:12, 0:3] = np.eye(3)  # 新增：角速度直接受控制 / Add: Angular rates directly controlled
        self.B[11, 3] = 1.0  # yaw_rate control
        
        # 权重矩阵 / Weight matrices
        #self.Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # State weights
        #self.R = np.diag([0.1, 0.1, 0.1, 0.1])  # Control weights
        self.Q = np.diag([1, 1, 1,    # x,y,z
                 0.5, 0.5, 0.5,  # vx,vy,vz
                 0.1, 0.1, 0.1,  # roll,pitch,yaw
                 0.01, 0.01, 0.01])  # p,q,r
        self.R = np.diag([0.5, 0.5, 0.5, 0.5])  # vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd
        
        # 求解Riccati方程 / Solve Riccati equation
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P  # 反馈增益矩阵 / Feedback gain matrix (4x12)
        
        # 保留PID控制偏航角（LQR不直接处理偏航） / Keep PID for yaw (LQR doesn't handle yaw well)
        self.pid_yaw = PIDController(Kp=0.5, Ki=0.01, Kd=0.1, Ki_sat=[0.5, 0.5, 0.5])

    def controller(self, state, target_pos, dt):
        """
        LQR主控制函数 / Main LQR control function
        输入 / Inputs:
            - state: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r] (12维状态 / 12D state)
            - target_pos: [x_target, y_target, z_target, yaw_target] (目标位置和偏航角 / Target position and yaw)
            - dt: 时间步长 / Time step (s)
        输出 / Outputs:
            - (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd): 速度指令 / Velocity commands
        """
        # 提取状态 / Extract state
        x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r = state
        x_target, y_target, z_target, yaw_target = target_pos
        
        # 计算状态误差 / Compute state error
        error = np.array([
            x_target - x,
            y_target - y,
            z_target - z,
            0 - vx,  # 目标速度通常为0 / Target velocity usually 0
            0 - vy,
            0 - vz,
            0 - roll,  # 目标姿态角为0（水平） / Target attitude 0 (level)
            0 - pitch,
            0 - yaw,  # 偏航角由PID单独处理 / Yaw handled separately by PID
            0 - p,
            0 - q,
            0 - r
        ])
        
        # LQR控制律 / LQR control law: u = -K * error
        u = -self.K @ error
        
        # 提取速度指令并限幅 / Extract velocity commands and clip
        vx_cmd = np.clip(u[0], -1, 1)
        vy_cmd = np.clip(u[1], -1, 1)
        vz_cmd = np.clip(u[2], -1, 1)
        
        # PID控制偏航角 / PID for yaw
        yaw_error = yaw_target - yaw
        yaw_rate_cmd = self.pid_yaw.control_update(np.array([yaw_error, 0, 0]), dt)[0]
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.74533, 1.74533)  # ±100°/s
        
        return (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)

# 全局实例化 / Global instance
lqr_controller = LQRController()

"""
def controller(state, target_pos, dt):
    # 提供给仿真器的接口函数 / Interface function for simulator
    # 注意：仿真器提供的state是6维，需扩展为12维 / Note: Simulator provides 6D state, expand to 12D

    full_state = np.concatenate([
        state[0:3],  # Position
        [0, 0, 0],   # Placeholder for velocity (not provided by simulator)
        state[3:6],  # Attitude
        [0, 0, 0]    # Placeholder for angular rates
    ])
    return lqr_controller.controller(full_state, target_pos, dt)
"""
def controller(state, target_pos, dt):
    return cascade_controller.controller(state, target_pos, dt)