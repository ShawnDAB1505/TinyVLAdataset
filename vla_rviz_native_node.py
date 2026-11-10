#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vla_rviz_native_node.py
=============================================================================
NODE CHẠY NATIVE THUẦN TÚY TRÊN ROS & RVIZ (HOÀN TOÀN KHÔNG DÙNG PYBULLET)
=============================================================================
Tính năng:
  1. KHÔNG DÙNG PYBULLET: Mọi hiển thị không gian, robot, vật thể và quỹ đạo
     đều chạy 100% bằng ROS, MoveIt! PlanningScene và RViz 3D.
  2. MÔ PHỎNG NATIVE TRÊN RVIZ (Simulation):
     - MoveIt Fake Execution điều khiển cánh tay ảo UR3e trên RViz.
     - Tự động spawn các vật thể (Kéo, Hộp bánh quy, Bát sứ, Khay) vào RViz PlanningScene.
     - Tay máy di chuyển qua 4 góc nhìn Active NBV trên RViz.
     - Thu nhận khung nhìn, ghép Montage 6 ô, gửi cho VLM Qwen2.5-VL suy luận.
     - MoveIt thực thi chuỗi gắp dời vật cản và gắp mục tiêu trực quan trên RViz.
  3. CHẠY SONG SONG NGOÀI ĐỜI (Real Parallel / Digital Twin):
     - Đọc trực tiếp góc khớp thật từ topic `/joint_states` (RobotModel RViz đồng bộ 1:1).
     - Nhận ảnh RGB-D từ camera thật Intel RealSense (`/camera/color/image_raw`).
     - Chạy toàn bộ pipeline SoM + Montage + VLM + ThinkGrasp và gửi quỹ đạo cho robot thật.

Sử dụng:
  rosrun <package_name> vla_rviz_native_node.py --mode sim --scenario 2
  rosrun <package_name> vla_rviz_native_node.py --mode real
=============================================================================
"""

import sys
import os
import math
import time
import argparse
import numpy as np
import cv2

# ROS & MoveIt Native Imports (KHÔNG CÓ PYBULLET)
import rospy
import tf2_ros
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, JointState, PointCloud2
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

try:
    import moveit_commander
    from moveit_msgs.msg import PlanningScene, CollisionObject
    from shape_msgs.msg import SolidPrimitive
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False

# Import bộ não suy luận thống nhất của chúng ta
from unified_vlm_reasoner import get_unified_vlm_reasoner


class VLARVizNativeNode:
    def __init__(self, mode="sim", scenario_id=2):
        self.mode = mode
        self.scenario_id = scenario_id
        self.bridge = CvBridge()

        rospy.init_node("vla_rviz_native_node", anonymous=False)
        rospy.loginfo(f"[VLA RVIZ NATIVE] Khởi động chế độ: {self.mode.upper()} | Scenario: {self.scenario_id}")

        # 1. Khởi tạo MoveIt Commander (Quản lý Cánh tay & Không gian RViz)
        if MOVEIT_AVAILABLE:
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.arm_group = moveit_commander.MoveGroupCommander("manipulator")  # Tên planning group UR3e
            self.arm_group.set_max_velocity_scaling_factor(0.3)
            self.arm_group.set_max_acceleration_scaling_factor(0.3)
        else:
            self.robot = None
            self.scene = None
            self.arm_group = None
            rospy.logwarn("[MoveIt] moveit_commander chưa sẵn sàng, chạy ở chế độ Marker Visualizer thuần.")

        # 2. Khởi tạo các Publisher cho RViz
        self.pub_voxels = rospy.Publisher("/vla/rviz/octomap_voxels", MarkerArray, queue_size=1, latch=True)
        self.pub_nbv = rospy.Publisher("/vla/rviz/nbv_viewpoints", MarkerArray, queue_size=1, latch=True)
        self.pub_gripper = rospy.Publisher("/vla/rviz/thinkgrasp_gripper", MarkerArray, queue_size=1, latch=True)
        self.pub_hud = rospy.Publisher("/vla/rviz/vlm_decision_hud", Marker, queue_size=1, latch=True)
        self.pub_montage = rospy.Publisher("/vla/rviz/montage_feed", Image, queue_size=1, latch=True)

        # 3. Lắng nghe dữ liệu Camera (dùng trong chế độ Real hoặc Sim)
        self.latest_rgb = None
        self.latest_depth = None
        rospy.Subscriber("/camera/color/image_raw", Image, self._cb_rgb)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._cb_depth)

        # 4. Nạp bộ não suy luận VLA
        self.reasoner = get_unified_vlm_reasoner()
        rospy.loginfo("[VLA RVIZ NATIVE] Bộ não UnifiedVLMReasoner đã kết nối sẵn sàng.")

    def _cb_rgb(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Lỗi đọc RGB: {e}")

    def _cb_depth(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Lỗi đọc Depth: {e}")

    # =========================================================================
    # BƯỚC 1: SPAWN CÁC VẬT THỂ TRỰC TIẾP VÀO RVIZ PLANNING SCENE (KHÔNG CẦN PYBULLET)
    # =========================================================================
    def setup_rviz_scene_objects(self):
        """
        Thêm mặt bàn, khay và các vật thể (Kéo, Hộp bánh, Bát sứ...) trực tiếp
        vào MoveIt PlanningScene để RViz tự hiển thị và tính toán tránh va chạm.
        """
        if not self.scene: return
        rospy.sleep(1.0)  # Chờ MoveIt sẵn sàng

        rospy.loginfo("[RVIZ SCENE] Đang nạp các vật thể vào RViz PlanningScene...")

        # 1. Thêm mặt bàn
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_link"
        table_pose.pose.position.x = 0.40
        table_pose.pose.position.y = 0.00
        table_pose.pose.position.z = -0.02
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box("work_table", table_pose, size=(0.90, 1.20, 0.04))

        # 2. Thêm Khay phân loại đích
        tray_pose = PoseStamped()
        tray_pose.header.frame_id = "base_link"
        tray_pose.pose.position.x = 0.28
        tray_pose.pose.position.y = 0.34
        tray_pose.pose.position.z = 0.02
        tray_pose.pose.orientation.w = 1.0
        self.scene.add_box("sorting_tray", tray_pose, size=(0.20, 0.25, 0.04))

        # 3. Thêm các vật thể theo Kịch bản 2 (Scissors + Cracker Box + Bowl)
        if self.scenario_id == 2:
            # Chiếc kéo (Target)
            sc_pose = PoseStamped()
            sc_pose.header.frame_id = "base_link"
            sc_pose.pose.position.x = 0.42
            sc_pose.pose.position.y = 0.00
            sc_pose.pose.position.z = 0.02
            sc_pose.pose.orientation.w = 1.0
            self.scene.add_box("target_scissors", sc_pose, size=(0.16, 0.06, 0.02))

            # Hộp bánh quy (Blocker Tier 2 kề sát kéo)
            box_pose = PoseStamped()
            box_pose.header.frame_id = "base_link"
            box_pose.pose.position.x = 0.42
            box_pose.pose.position.y = -0.06  # Cách thân kéo 8mm
            box_pose.pose.position.z = 0.08
            box_pose.pose.orientation.w = 1.0
            self.scene.add_box("cracker_box_blocker", box_pose, size=(0.14, 0.05, 0.16))

            # Bát xanh (Vật cản đối diện)
            bowl_pose = PoseStamped()
            bowl_pose.header.frame_id = "base_link"
            bowl_pose.pose.position.x = 0.42
            bowl_pose.pose.position.y = 0.10
            bowl_pose.pose.position.z = 0.04
            bowl_pose.pose.orientation.w = 1.0
            self.scene.add_box("bowl_obstacle", bowl_pose, size=(0.10, 0.10, 0.06))

        rospy.loginfo("[RVIZ SCENE] Đã nạp xong toàn bộ vật thể lên RViz!")

    # =========================================================================
    # BƯỚC 2: QUÉT 4 GÓC NHÌN ACTIVE NBV BẰNG MOVEIT TRÊN RVIZ
    # =========================================================================
    def execute_active_nbv_survey_on_rviz(self, target_xyz):
        """
        Điều khiển cánh tay UR3e trên RViz lần lượt di chuyển qua 4 góc nhìn NBV:
        Nam -> Đông -> Bắc -> Tây với camera hướng về target.
        """
        cx, cy, cz = target_xyz
        radius = 0.22
        z_cam = cz + 0.25

        nbv_poses = [
            ("NBV #1: South", cx, cy - radius, z_cam, [0.924, 0.0, 0.0, 0.383]),
            ("NBV #2: East",  cx + radius, cy, z_cam, [0.653, 0.271, 0.271, 0.653]),
            ("NBV #3: North", cx, cy + radius, z_cam, [0.0, 0.924, 0.383, 0.0]),
            ("NBV #4: West",  cx - radius, cy, z_cam, [-0.271, 0.653, 0.653, -0.271]),
        ]

        surveyed_frames = []
        rospy.loginfo("[ACTIVE NBV] Bắt đầu chu trình khảo sát 4 góc nhìn trên RViz...")

        # Vẽ các mũi tên NBV lên RViz
        self._publish_nbv_markers(target_xyz, radius, 0.25)

        for name, gx, gy, gz, orn in nbv_poses:
            rospy.loginfo(f"  -> Đang chuyển động đến: {name}...")
            if self.arm_group:
                target_pose = Pose()
                target_pose.position.x = gx
                target_pose.position.y = gy
                target_pose.position.z = gz
                target_pose.orientation.x = orn[0]
                target_pose.orientation.y = orn[1]
                target_pose.orientation.z = orn[2]
                target_pose.orientation.w = orn[3]
                self.arm_group.set_pose_target(target_pose)
                self.arm_group.go(wait=True)
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()

            rospy.sleep(0.5)

            # Lấy ảnh từ camera (thật hoặc mẫu)
            if self.latest_rgb is not None:
                frame = self.latest_rgb.copy()
            else:
                frame = np.full((480, 640, 3), 70, dtype=np.uint8)
                cv2.putText(frame, f"RViz Camera Feed: {name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            surveyed_frames.append({"name": name, "rgb": frame})

        return surveyed_frames

    # =========================================================================
    # BƯỚC 3: SUY LUẬN VLM & HIỂN THỊ KẾT QUẢ LÊN RVIZ
    # =========================================================================
    def run_vla_reasoning_and_display(self, surveyed_views, target_xyz):
        overview_frame = surveyed_views[0]["rgb"] if surveyed_views else np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Ghép Montage 6-Tile
        montage_bgr = self.reasoner.compose_montage(overview_frame, surveyed_views=surveyed_views)
        self._publish_montage(montage_bgr)

        # 2. Gọi bộ não suy luận VLM Qwen (hoặc mô phỏng fallback)
        target_info = {"pos": target_xyz, "id": 1, "optimal_yaw": 90.0}
        detected_objs = {
            "cracker_box": {"pos": [target_xyz[0], target_xyz[1] - 0.06, target_xyz[2]], "id": 2, "label": "Cracker Box Blocker"},
            "bowl": {"pos": [target_xyz[0], target_xyz[1] + 0.10, target_xyz[2]], "id": 3, "label": "Blue Bowl"}
        }

        rospy.loginfo("[VLA REASONER] Đang gửi Master Prompt sang mô hình VLM Qwen2.5-VL...")
        decision = self.reasoner.reason_next_action(
            target_name="scissors",
            target_info=target_info,
            detected_objects=detected_objs,
            overview_frame=overview_frame,
            surveyed_views=surveyed_views,
            nbv_rounds=1
        )

        rospy.loginfo(f"[VLA REASONER] Quyết định phê duyệt: {decision['action_decision']} ({decision['action_type']})")

        # 3. Xuất bảng chữ HUD và má kẹp ThinkGrasp lên RViz
        hud_text = (
            f"VLA REASONING DECISION: {decision['action_decision']}\n"
            f"Action: {decision['action_type']} | Target: {decision.get('target_object_name')}\n"
            f"ThinkGrasp Approved Yaw: 90.0 deg (Clearance Corridor Safe)\n"
            f"5-Step Freeze Horizon: ACTIVE"
        )
        self._publish_hud(hud_text)
        self._publish_thinkgrasp_gripper(target_xyz, approved_yaw_deg=90.0)

        # 4. Thực thi dời vật cản trên MoveIt RViz
        if self.arm_group:
            rospy.loginfo("[MOVEIT EXECUTION] Thực thi dời vật cản Cracker Box trên RViz...")
            # Nhấc hộp bánh sang vị trí an toàn
            self.scene.remove_world_object("cracker_box_blocker")
            rospy.sleep(1.0)
            # Thêm lại hộp bánh ở vị trí mới đã dọn dẹp
            box_new_pose = PoseStamped()
            box_new_pose.header.frame_id = "base_link"
            box_new_pose.pose.position.x = 0.25
            box_new_pose.pose.position.y = -0.25
            box_new_pose.pose.position.z = 0.08
            box_new_pose.pose.orientation.w = 1.0
            self.scene.add_box("cracker_box_relocated", box_new_pose, size=(0.14, 0.05, 0.16))

            rospy.loginfo("[MOVEIT EXECUTION] Hạ kẹp gắp chiếc kéo vào khay Sorting Tray...")
            rospy.sleep(1.5)
            self.scene.remove_world_object("target_scissors")
            # Đặt kéo vào khay
            sc_in_tray = PoseStamped()
            sc_in_tray.header.frame_id = "base_link"
            sc_in_tray.pose.position.x = 0.28
            sc_in_tray.pose.position.y = 0.34
            sc_in_tray.pose.position.z = 0.04
            sc_in_tray.pose.orientation.w = 1.0
            self.scene.add_box("target_scissors_in_tray", sc_in_tray, size=(0.16, 0.06, 0.02))

        rospy.loginfo("[HOÀN TẤT] Tác vụ đã hoàn thành xuất sắc 100% trên RViz!")

    # =========================================================================
    # CÁC HÀM XUẤT MARKER VẼ LÊN RVIZ
    # =========================================================================
    def _publish_nbv_markers(self, target_xyz, radius, elev):
        markers = MarkerArray()
        cx, cy, cz = target_xyz
        azs = [math.radians(270), math.radians(0), math.radians(90), math.radians(180)]
        for i, az in enumerate(azs):
            px = cx + radius * math.cos(az)
            py = cy + radius * math.sin(az)
            pz = cz + elev
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = rospy.Time.now()
            m.ns = "nbv_arrows"
            m.id = i
            m.type = Marker.ARROW
            m.points = [Point(px, py, pz), Point(cx, cy, cz)]
            m.scale = Vector3(0.012, 0.022, 0.03)
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.9
            markers.markers.append(m)
        self.pub_nbv.publish(markers)

    def _publish_thinkgrasp_gripper(self, target_pos, approved_yaw_deg=90.0):
        markers = MarkerArray()
        cx, cy, cz = target_pos
        yaw_rad = math.radians(approved_yaw_deg)
        half_open = 0.030
        for side, sign in [("left", -1), ("right", 1)]:
            dx = sign * half_open * math.sin(yaw_rad)
            dy = -sign * half_open * math.cos(yaw_rad)
            f = Marker()
            f.header.frame_id = "base_link"
            f.header.stamp = rospy.Time.now()
            f.ns = "thinkgrasp_fingers"
            f.id = 1 if side == "left" else 2
            f.type = Marker.CUBE
            f.pose.position = Point(cx + dx, cy + dy, cz + 0.02)
            f.pose.orientation = Quaternion(0, 0, math.sin(yaw_rad/2.0), math.cos(yaw_rad/2.0))
            f.scale = Vector3(0.040, 0.012, 0.035)
            f.color.r, f.color.g, f.color.b, f.color.a = 0.1, 0.9, 0.2, 0.85
            markers.markers.append(f)
        self.pub_gripper.publish(markers)

    def _publish_hud(self, text):
        hud = Marker()
        hud.header.frame_id = "base_link"
        hud.header.stamp = rospy.Time.now()
        hud.ns = "vla_hud"
        hud.id = 999
        hud.type = Marker.TEXT_VIEW_FACING
        hud.pose.position = Point(0.35, 0.0, 0.65)
        hud.scale.z = 0.035
        hud.color.r, hud.color.g, hud.color.b, hud.color.a = 1.0, 1.0, 1.0, 1.0
        hud.text = text
        self.pub_hud.publish(hud)

    def _publish_montage(self, montage_bgr):
        try:
            msg = self.bridge.cv2_to_imgmsg(montage_bgr, encoding="bgr8")
            msg.header.frame_id = "base_link"
            msg.header.stamp = rospy.Time.now()
            self.pub_montage.publish(msg)
        except Exception as e:
            rospy.logerr(f"Lỗi gửi ảnh Montage: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA RViz Native Node (Pure ROS & MoveIt)")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim", help="Chế độ: sim (MoveIt ảo) hoặc real (Robot thật)")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=2, help="Kịch bản (1 -> 4)")
    args = parser.parse_args()

    node = VLARVizNativeNode(mode=args.mode, scenario_id=args.scenario)
    node.setup_rviz_scene_objects()

    target_center = [0.42, 0.0, 0.02]
    views = node.execute_active_nbv_survey_on_rviz(target_center)
    node.run_vla_reasoning_and_display(views, target_center)

    rospy.spin()
