====================================================================================================
      HƯỚNG DẪN CÀI ĐẶT DEPENDENCIES VÀ CHẠY PIPELINE NATIVE THUẦN TÚY TRÊN UBUNTU RVIZ 
                    (HOÀN TOÀN KHÔNG DÙNG PYBULLET - CHẠY 100% BẰNG ROS & RVIZ)
====================================================================================================

Tài liệu này hướng dẫn:
  1. DANH SÁCH FILE CẦN ĐĂNG TẢI / COPY SANG MÁY UBUNTU (Gọn nhẹ, không cần PyBullet).
  2. Cài đặt các thư viện (Dependencies) của Pipeline.
  3. Cách chạy Node ROS Native (`vla_rviz_native_node.py`) để toàn bộ không gian 3D, cánh tay robot,
     vật thể, camera, và quá trình suy luận VLM hiển thị trực tiếp trên giao diện RVIZ.
  4. Hai chế độ vận hành:
     - CHẾ ĐỘ 1: MÔ PHỎNG NATIVE TRÊN RVIZ (Simulation Mode) - Dùng MoveIt Fake Controller.
     - CHẾ ĐỘ 2: CHẠY SONG SONG NGOÀI ĐỜI (Real Parallel Mode) - Lấy ảnh từ camera RealSense thật, 
       chạy pipeline VLA và chiếu trực tiếp kết quả lên RViz theo thời gian thực.

====================================================================================================
PHẦN 1: DANH SÁCH CÁC FILE CẦN ĐĂNG TẢI SANG MÁY UBUNTU (FILE MANIFEST)
====================================================================================================

Bạn chỉ cần nén và copy đúng các file sau sang Ubuntu (đặt vào thư mục `~/vla_pipeline`):

1. MÃ NGUỒN PIPELINE VÀ NODE RVIZ NATIVE:
   ├── vla_rviz_native_node.py        <- Node ROS/MoveIt Native (Chạy trực tiếp trên RViz, KHÔNG DÙNG PyBullet)
   ├── unified_vlm_reasoner.py        <- Bộ não suy luận trung tâm (VLM Qwen, KnowNo Conformal Prediction, 5-Step Freeze)
   ├── benchmark_profiler.py          <- Module đo đạc thời gian profiler (T_VLM, T_motion, T_perc)
   ├── graspnet_policy_adapter.py     <- Bộ giải pháp gắp GraspNet / Heuristic
   ├── ycb_objects.py                 <- Quản lý thông tin hình học các vật thể
   ├── requirements_pipeline_ubuntu.txt <- Danh sách thư viện Python cần cài đặt
   └── HUONG_DAN_CHAY_PIPELINE_RVIZ.txt <- Tài liệu hướng dẫn này

(HOÀN TOÀN KHÔNG CẦN copy các file mô phỏng pybullet hay cài pybullet trên Ubuntu!)

====================================================================================================
PHẦN 2: CÀI ĐẶT DEPENDENCIES TRÊN UBUNTU (MỘT LẦN DUY NHẤT)
====================================================================================================

Mở Terminal trên Ubuntu và chạy các lệnh sau:

--- BƯỚC 2.1: CÀI CÁC GÓI ROS & MOVEIT ---
sudo apt update
sudo apt install -y python3-pip python3-dev git curl
sudo apt install -y \
    ros-noetic-rviz \
    ros-noetic-moveit \
    ros-noetic-moveit-commander \
    ros-noetic-cv-bridge \
    ros-noetic-visualization-msgs

--- BƯỚC 2.2: CÀI CÁC THƯ VIỆN PYTHON (PIP) ---
cd ~/vla_pipeline
pip3 install -r requirements_pipeline_ubuntu.txt

# (Khuyến nghị nếu máy có GPU NVIDIA):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

--- BƯỚC 2.3: CÀI OLLAMA VÀ TẢI MODEL VLM QWEN2.5-VL ---
curl -fsSL https://ollama.com/install.sh | sh
ollama run qwen2.5vl:7b
# Khi model tải xong và hiện dấu nhắc >>> gõ: /bye để thoát ra.
# Ollama sẽ tự động duy trì chạy nền tại cổng http://localhost:11434.

====================================================================================================
PHẦN 3: CẤU HÌNH GIAO DIỆN RVIZ ĐỂ HIỂN THỊ NATIVE PIPELINE
====================================================================================================

1. Mở Terminal 1: Khởi động MoveIt và RViz (ở chế độ Fake Controller):
   roslaunch ur3e_moveit_config demo.launch
   (RViz sẽ mở lên với cánh tay UR3e 3D ở tư thế mặc định).

2. Cấu hình các Plugin hiển thị trên cửa sổ RViz:
   - Trong mục "Displays" bên trái màn hình:
     * Đã có sẵn `RobotModel` và `MotionPlanning` (hiển thị cánh tay robot và planning scene).
   - Bấm nút "Add" (ở góc dưới bên trái) để thêm các hiển thị của Pipeline:
     * Add -> `MarkerArray` -> Đổi tên thành: `Active NBV Viewpoints`
       -> Đặt Topic: `/vla/rviz/nbv_viewpoints`
     * Add -> `MarkerArray` -> Đổi tên thành: `ThinkGrasp Gripper`
       -> Đặt Topic: `/vla/rviz/thinkgrasp_gripper`
     * Add -> `Marker`      -> Đổi tên thành: `VLM Decision HUD`
       -> Đặt Topic: `/vla/rviz/vlm_decision_hud`
     * Add -> `Image`       -> Đổi tên thành: `Set-of-Mark Montage Feed`
       -> Đặt Topic: `/vla/rviz/montage_feed`

3. Lưu cấu hình: File -> Save Config As -> `~/vla_pipeline/vla_rviz.rviz`.

====================================================================================================
PHẦN 4: CHẠY CHẾ ĐỘ 1: MÔ PHỎNG NATIVE TRÊN RVIZ (SIMULATION MODE)
====================================================================================================

Trong chế độ này, node `vla_rviz_native_node.py` tự động spawn các vật thể vào RViz PlanningScene,
điều khiển cánh tay UR3e trên RViz qua 4 góc Active NBV, gửi ảnh sang VLM và thực thi gắp!

1. Mở Terminal 1: Khởi động RViz
   roslaunch ur3e_moveit_config demo.launch

2. Mở Terminal 2: Chạy Node VLA Native (Kịch bản 2)
   cd ~/vla_pipeline
   python3 vla_rviz_native_node.py --mode sim --scenario 2

* QUÁ TRÌNH DIỄN RA TRÊN MÀN HÌNH RVIZ:
  1. Mặt bàn, khay phân loại, chiếc kéo (Target) và hộp bánh quy (Blocker) xuất hiện ngay trên RViz.
  2. Cánh tay UR3e trên RViz tự động di chuyển lần lượt qua 4 góc nhìn Active NBV quanh chiếc kéo.
  3. Bốn mũi tên xanh lơ xuất hiện trong không gian RViz đánh dấu các vị trí camera đã chụp.
  4. Montage 6-Tile hiển thị trên cửa sổ Image của RViz.
  5. VLM Ollama đưa ra quyết định: "Option B (Relocate Blocker) | Approved Yaw: 90 deg".
  6. Má kẹp ThinkGrasp xuất hiện trên RViz (màu xanh lá) kiểm tra hành lang kẹp.
  7. Cánh tay UR3e trên RViz thực thi dời hộp bánh sang bên hông, sau đó hạ kẹp gắp kéo vào khay!

====================================================================================================
PHẦN 5: CHẠY CHẾ ĐỘ 2: SONG SONG NGOÀI ĐỜI (REAL SENSOR / PARALLEL MODE)
====================================================================================================

Trong chế độ này:
  * Camera Intel RealSense thật quay các vật thể thật trên bàn ngoài đời.
  * Pipeline nhận ảnh thật -> chạy YOLOE SoM -> gọi VLM Qwen2.5-VL -> tính ThinkGrasp.
  * RViz hiển thị Digital Twin thời gian thực và chiếu kết quả xử lý lên màn hình 3D!

--- CÁC LỆNH CHẠY ---

1. Cắm camera RealSense D435 vào cổng USB 3.0 máy Ubuntu.

2. Mở Terminal 1: Khởi động camera RealSense
   roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud

3. Mở Terminal 2: Mở RViz
   roslaunch ur3e_moveit_config demo.launch

4. Mở Terminal 3: Chạy Node VLA ở chế độ Real Sensor
   cd ~/vla_pipeline
   python3 vla_rviz_native_node.py --mode real --scenario 2

* KẾT QUẢ QUAN SÁT:
  - Khi bạn đặt chiếc kéo và hộp bánh ngoài đời thật, camera chụp ảnh gửi vào pipeline.
  - VLM nhận thức hiện trường thật, tính toán góc kẹp an toàn và hiển thị trực tiếp quyết định lên RViz!
====================================================================================================
