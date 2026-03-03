# References:
# - https://github.com/jimmyyhwu/tidybot2/blob/main/ik_solver.py
# - https://github.com/bulletphysics/bullet3/blob/master/examples/ThirdPartyLibs/BussIK/Jacobian.cpp
# - https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
# - https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py

import mujoco
import numpy as np
# from constants import RESET_CMD

DAMPING_COEFF = 1e-12
MAX_ANGLE_CHANGE = np.deg2rad(45)


class IKSolver:
    def __init__(self, calibrate=False):
        # Load arm without gripper
        self.model = mujoco.MjModel.from_xml_path("models/kinova_gen3/gen3_ours.xml")
        if calibrate:
            self.model = mujoco.MjModel.from_xml_path(
                "models/kinova_gen3/gen3_calib.xml"
            )

        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        # self.qpos0 = self.model.key("retract").qpos
        # self.qpos0 = RESET_CMD
        self.site_id = self.model.site("pinch_site").id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)
        self.eye = np.eye(self.model.nv)

    def unit_vec_from_quat(self, q, axis):
        """
        Compute the unit vector corresponding to the specified axis
        from the given quaternion.
        """
        rot = np.zeros(shape=(9,), dtype=np.float64)
        mujoco.mju_quat2Mat(rot, q)
        rot = rot.reshape((3, 3))
        unit_vec = rot[:, axis]
        return unit_vec

    def run_fk(self, qpos):
        """
        Forward-kinematics convenience wrapper.

        Args
        ----
        qpos : (self.model.nq,) array_like
            Joint angles (radians) in the model’s generalized‐coordinate order.

        Returns
        -------
        pos  : (3,)  np.ndarray   – world-frame XYZ of the end-effector site
        quat : (4,)  np.ndarray   – orientation quaternion (x, y, z, w)
        """
        # Load the joint configuration
        self.data.qpos[:] = qpos

        # Update kinematics (positions, orientations, etc.)
        mujoco.mj_kinematics(self.model, self.data)

        # Copy out world-space position
        pos = self.site_pos.copy()

        # Convert the site’s rotation matrix to quaternion
        mujoco.mju_mat2Quat(self.site_quat, self.site_mat)  # returns (w, x, y, z)

        # Re-order to (x, y, z, w) to match the rest of your codebase
        quat = self.site_quat[[1, 2, 3, 0]].copy()

        return pos, quat

    def solve2(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        quat = quat[[3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)

        # Set arm to initial joint configuration
        self.data.qpos = curr_qpos

        # cache for nullspace bias reference
        q_start = curr_qpos.copy()

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            # Calculate update
            mujoco.mj_jacSite(
                self.model, self.data, self.jac_pos, self.jac_rot, self.site_id
            )
            update = self.jac.T @ np.linalg.solve(
                self.jac @ self.jac.T + self.damping, self.err
            )
            # qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            qpos0_err = np.mod(q_start - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (
                self.eye
                - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping))
                @ self.jac
            ) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > MAX_ANGLE_CHANGE:
                update *= MAX_ANGLE_CHANGE / update_max

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()

    def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        quat = quat[[3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)

        # Set arm to initial joint configuration
        self.data.qpos = curr_qpos

        # cache for nullspace bias reference
        q_start = curr_qpos.copy()

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            mujoco.mj_jacSite(
                self.model, self.data, self.jac_pos, self.jac_rot, self.site_id
            )
            J = self.jac  # 6x7, world frame (rows: [lin; ang])

            # (A) Task row-weighting: make some task components more important.
            S = np.diag(
                [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
            )  # pos xyz, then ori rpy small-angle
            Jt = S @ J  # 6x7
            err_t = S @ self.err  # 6

            # (B) Adaptive joint weights (bigger number = penalize more)
            Jp = self.jac_pos  # 3x7 (linear block)

            # desired translation direction u (when mostly translating)
            e = self.err_pos
            u = (
                e / (np.linalg.norm(e) + 1e-12)
                if np.linalg.norm(e) > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )
            P_orth = np.eye(3) - np.outer(u, u)

            mag = np.linalg.norm(Jp, axis=0)  # overall leverage
            off_axis = np.linalg.norm(P_orth @ Jp, axis=0)  # motion not along u

            # combine (tune α,β as needed)
            alpha, beta = 1.0, 0.0
            # alpha, beta = 0.0, 1.0
            raw_w = alpha * mag + beta * off_axis

            # normalize/clamp; bias to favor distal joints
            raw_w /= np.median(raw_w) + 1e-12
            raw_w = np.clip(raw_w, 0.5, 6.0)

            # favors distal joints (wrists) to reduce Cartesian movement
            # static_bias = np.array([2.0, 1.6, 1.6, 1.0, 0.9, 0.8, 0.7])  # J1..J7
            static_bias = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # J1..J7
            W_diag = raw_w * static_bias
            Winv = np.diag(1.0 / W_diag)

            # (C) Damping adapted by manipulability (keeps solver well-conditioned near singularities)
            m = np.sqrt(np.linalg.det(J @ J.T) + 1e-12)  # Yoshikawa manipulability
            lam_min, lam_max = 1e-4, 5e-2
            lam = np.clip(
                lam_max - (lam_max - lam_min) * np.tanh(10.0 * m), lam_min, lam_max
            )

            # (D) Weighted pseudoinverse (first-order exact on the weighted task)
            # dq_task = Winv * Jt^T * (Jt * Winv * Jt^T + lam^2 I)^-1 * err_t
            A = Jt @ Winv @ Jt.T + (lam**2) * np.eye(6)
            dq_task = Winv @ Jt.T @ np.linalg.solve(A, err_t)

            # (E) Weighted null-space posture term (toward q_start), small gain so it won't inject wobble
            Jsharp = Winv @ Jt.T @ np.linalg.solve(A, np.eye(6))  # 7x6
            N = np.eye(self.model.nv) - Jsharp @ Jt
            qpos_err = np.mod(q_start - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            posture_gain = 0.05
            update = dq_task + posture_gain * (N @ qpos_err)

            # (F) Per-joint clamp to keep linearization valid (e.g., ≤ 2° per iteration)
            max_step = np.deg2rad(2.0)
            update = np.clip(update, -max_step, +max_step)

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        # if np.linalg.norm(self.err) < err_thresh:
        #     print(f"IK solved in {_} iterations")
        # else:
        #     print(f"IK failed to solve in {_} iterations")

        return self.data.qpos.copy()


if __name__ == "__main__":
    ik_solver = IKSolver()
    reset_qpos = np.deg2rad(
        np.array([14.66, 20.41, 339.92, 95.79, 7.08, 63.03, 257.29])
    )
    # Test IK accuracy
    pos0, quat0 = ik_solver.run_fk(reset_qpos)
    print(f"FK test [home]: pos={pos0.round(3)}, quat={quat0.round(3)}")
    n_steps = 20
    dy_step = 0.005  # 0.5cm

    qpos_diff = []
    qpos_diff2 = []

    def run_one_dir(sign):
        q = reset_qpos.copy()
        q2 = reset_qpos.copy()
        pos = pos0.copy()
        pos2 = pos0.copy()

        for k in range(n_steps):
            # desired: +y only
            target_pos = pos + np.array([0.0, sign * dy_step, 0.0])

            # one IK step from current q
            q_next = ik_solver.solve(target_pos, quat0, q)

            q_next2 = ik_solver.solve2(target_pos, quat0, q)

            dq = q_next - q
            qpos_diff.append(dq)

            dq2 = q_next2 - q
            qpos_diff2.append(dq2)

            pos_next, _ = ik_solver.run_fk(q_next)
            pos_next2, _ = ik_solver.run_fk(q_next2)

            # advance
            q, pos = q_next, pos_next
            q2, pos2 = q_next2, pos_next2

    for s in [+1, -1]:
        run_one_dir(s)

    # summarize qpos_diff
    qpos_diff = np.array(qpos_diff)
    qpos_diff_deg = np.rad2deg(qpos_diff)

    qpos_diff2 = np.array(qpos_diff2)
    qpos_diff2_deg = np.rad2deg(qpos_diff2)

    # joint statistics
    for i in range(qpos_diff.shape[1]):
        print(f"joint {i}:")

        print(
            f"mean: {np.mean(qpos_diff_deg[:, i])}, std: {np.std(qpos_diff_deg[:, i])}"
        )
        print(
            f"mean2: {np.mean(qpos_diff2_deg[:, i])}, std2: {np.std(qpos_diff2_deg[:, i])}"
        )
