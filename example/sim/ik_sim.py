#!/usr/bin/env python3
"""逆运动学仿真 — 交互式输入目标位姿 + MeshCat 实时可视化。

用法:
    uv run python example/sim/ik_sim.py

控制:
    输入目标位置 x y z (米)
    可选: 姿态 roll pitch yaw (弧度)
    例: 0.25 0.0 0.25                    (仅位置)
    例: 0.29545 0.0 0.28664 0 0.17453 0  (位置+姿态)
    q / quit / exit: 退出
"""

import sys
import signal
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics.inverse_kinematics import compute_ik
from reBotArm_control_py.kinematics.robot_model import get_end_effector_frame_id
from reBotArm_control_py.kinematics.forward_kinematics import compute_fk
from example.sim.visualizer import Visualizer

should_exit = False


def signal_handler(sig, frame):
    global should_exit
    should_exit = True


def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("Loading visualizer...")
    viz = Visualizer()

    viz.neutral()
    q_seed = pin.neutral(viz.model)

    # gripper_end 挂在 joint6 后；只把该关节及其祖先作为机械臂关节显示，
    # 排除 URDF 中位于末端之后的两个夹爪滑动关节。
    end_frame = viz.model.frames[get_end_effector_frame_id(viz.model)]
    end_joint = viz.model.joints[end_frame.parentJoint]
    arm_nq = end_joint.idx_q + end_joint.nq

    print("MeshCat is open. Enter the target pose:")
    print("  x y z                      (position only, meters)")
    print("  x y z roll pitch yaw       (position+orientation, radians)")
    print("  Example reachable pose: 0.29545 0 0.28664 0 0.17453 0")
    print("  q/quit/exit: exit\n")

    while not should_exit:
        time.sleep(0.01)

        try:
            line = input("Target pose > ").strip().lower()
        except EOFError:
            break

        if line in ("q", "quit", "exit", ""):
            break

        try:
            vals = [float(x) for x in line.split()]
            if len(vals) not in (3, 6):
                print("Need 3 values (position only) or 6 values (position+orientation)\n")
                continue
        except ValueError:
            print("Invalid input\n")
            continue

        target_pos = np.array(vals[:3])  # 获取位置
        target_rot = None
        if len(vals) == 6:
            r, p, y = vals[3], vals[4], vals[5]
            target_rot = pin.rpy.rpyToMatrix(r, p, y)  # 获取姿态

        result = compute_ik(q_seed, target_pos, target_rot)

        if result.success:
            viz.update(result.q)
            # 后续目标从当前已到达构型继续求解，避免每次跳回零位。
            q_seed = result.q.copy()
        status = "Converged" if result.success else "Not converged"
        actual_pos, actual_rot, _ = compute_fk(viz.model, result.q)
        position_error = float(np.linalg.norm(target_pos - actual_pos))

        if target_rot is None:
            print(
                f"  [{status}] Iterations={result.iterations} "
                f"Position error={position_error:.2e}m"
            )
        else:
            orientation_error = float(np.linalg.norm(pin.log3(actual_rot.T @ target_rot)))
            print(
                f"  [{status}] Iterations={result.iterations} "
                f"Position error={position_error:.2e}m "
                f"Orientation error={orientation_error:.2e}rad"
            )
        print(f"  Arm joint angles (deg): {np.degrees(result.q[:arm_nq])}\n")

        if not result.success and target_rot is not None:
            position_result = compute_ik(q_seed, target_pos)
            if position_result.success:
                _, reachable_rot, _ = compute_fk(viz.model, position_result.q)
                reachable_rpy = pin.rpy.matrixToRpy(reachable_rot)
                print("  The position is reachable, but the requested orientation is unreachable within the current joint limits.")
                print("  For position-only control, enter just the first 3 values.")
                print(
                    "  Reachable RPY near the current position (rad): "
                    f"{np.array2string(reachable_rpy, precision=5)}\n"
                )


if __name__ == "__main__":
    main()
