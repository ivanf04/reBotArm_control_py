#!/usr/bin/env python3
"""
reBotArm 正运动学数据测试。
输入: model.nq 个关节角度，单位：度，空格分隔
输出: 末端位置 (x, y, z)，单位：米
      + 旋转矩阵 (3x3)
      + 欧拉角 (roll, pitch, yaw)，单位：度

Forward kinematics data test.
Input: model.nq joint angles in degrees, space-separated
Output: End-effector position (x, y, z) in meters
        + Rotation matrix (3x3)
        + Euler angles (roll, pitch, yaw) in degrees

用法 / Usage:
    python example/5_fk_test.py

配置 / Config: config/rebotarm.yaml
"""

import sys
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from reBotArm_control_py.kinematics import (
    get_joint_count,
    load_robot_model,
    compute_fk,
    get_joint_names,
)

# ----------------------------------------------------------------------
# 打印 / Print
# ----------------------------------------------------------------------
def print_welcome(model, joint_names) -> None:
    n = get_joint_count()
    print("=" * 52)
    print("  reBotArm Forward Kinematics Test")
    print("=" * 52)
    print(f"  Robot  : {model.name}")
    print(f"  Joints: {joint_names[:n]}")
    print(f"  nq = {model.nq} (URDF), controlling first {n} joints")
    print()
    print(f"  Enter {n} joint angles (deg), space-separated")
    print("-" * 52)
    print("> ", end="", flush=True)

def print_result(q_deg, position, rotation, euler_deg) -> None:
    print()
    print("=" * 52)
    print("  Result")
    print("=" * 52)
    print(f"  Joint angles (deg): {q_deg}")
    print()
    print(f"  End-effector position (m):")
    print(f"    X = {position[0]:+.6f}")
    print(f"    Y = {position[1]:+.6f}")
    print(f"    Z = {position[2]:+.6f}")
    print()
    print(f"  Rotation matrix (R_world^end):")
    for row in rotation:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print()
    print(f"  Euler XYZ (roll, pitch, yaw) [deg]:")
    print(f"    roll  = {euler_deg[0]:+.4f}")
    print(f"    pitch = {euler_deg[1]:+.4f}")
    print(f"    yaw   = {euler_deg[2]:+.4f}")

def parse_joint_input(line: str, n: int) -> np.ndarray:
    tokens = line.split()
    if len(tokens) != n:
        print(f"Error: need {n} values, got {len(tokens)}")
        sys.exit(1)
    try:
        q_deg = [float(x) for x in tokens]
    except ValueError as e:
        print(f"Error: cannot parse number — {e}")
        sys.exit(1)
    return np.radians(q_deg)


# ----------------------------------------------------------------------
# 核心算法 / Core algorithm
# ----------------------------------------------------------------------
def compute_fk_from_deg(model, q_deg: list) -> tuple:
    q_rad = np.radians(q_deg)
    full_q = np.zeros(model.nq)
    full_q[:len(q_rad)] = q_rad
    position, rotation, homogeneous = compute_fk(model, full_q)
    euler_deg = np.degrees(pin.rpy.matrixToRpy(rotation))
    return position, rotation, homogeneous, euler_deg

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    model = load_robot_model()
    joint_names = get_joint_names(model)
    n_joints = get_joint_count()

    print_welcome(model, joint_names)

    try:
        line = input().strip()
    except EOFError:
        print("No input, exiting.")
        return

    q_rad = parse_joint_input(line, n_joints)
    q_deg = np.degrees(q_rad)

    position, rotation, homogeneous, euler_deg = compute_fk_from_deg(model, q_deg)

    print_result(q_deg, position, rotation, euler_deg)

if __name__ == "__main__":
    main()
