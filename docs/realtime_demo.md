# 실시간 데모 커넥터 (`tools/realtime/`)

`tools/highlevel/scripts/run_realtime.py`가 만든 타임라인 JSON(`TimedRun`: 컨베이어가 움직이는 동안 계획, 이벤트마다 팔레트·버퍼·카메라 창 스냅샷)을 시각화 환경에서 시간 순서대로 재생합니다.

| 파일 | 역할 | 필요 환경 |
|---|---|---|
| `rt_timeline.py` | 타임라인 로드·정렬, 배치(팔레트 좌표) 추출, 컨베이어 창, 터미널 상태 줄, `Pacer`(타임라인 초 → 실제 초, `--speed`) | 표준 라이브러리 |
| `live_sim_bridge.py` | (A) 재성 AHEAD Live Physics Simulator(PyBullet + three.js, `http://127.0.0.1:4173`)의 HDR50-22 로봇 셀로 재생 | 표준 라이브러리(urllib) |
| `gazebo_replay.py` | (B) Gazebo HDR50-22 받침대 작업셀에서 재생: 관절 궤적 + 박스 생성/삭제 + 컨베이어 창 | ROS 2 Humble, `ros_gz_sim`, 팀 `pac_common`·`pac_robot_check`·`pac_runtime`(읽기 전용) |
| `export_viewer.py` | (C) 타임라인 → HTML 한 파일(three.js r128, cdnjs 고정 버전) | 브라우저 |
| `fixtures/timeline_sample.json` | 테스트용으로 줄인 타임라인(이벤트 0-5, 69-74, 팔레트 교체 포함) | - |

터미널 상태 줄(A, B 공통): `[i/N] t=시뮬시간 동작 박스 | view[n]: 카메라에 보이는 컨베이어 박스(id 크기) | compute s replans idle s | buffer | pallet 번호 fill %`, 팔레트 교체·재배치·지연(`behind schedule`)이 있으면 함께 표시합니다.

## 타임라인 만들기

```bash
cd ~/pac-mission1-lookahead
python3 tools/highlevel/scripts/run_realtime.py --dataset tools/highlevel/output/dataset80_x40_s8 \
    --split test --episode 0 --output reports/realtime_test0.json
```

## A. 재성 Live 시뮬레이터 브리지

API(`pac_simulation/ahead_sim/server.py`): `GET /api/state`(로봇 상태 `robot.state`, `queue`, `completed`, `log`), `POST /api/robot_place`(로봇이 PICK에서 흡착해 놓음), `POST /api/place`(로봇 없이 바로 놓기), `POST /api/reset`. 팔레트 교체 API는 없습니다.
좌표: 시뮬레이터 프레임은 팔레트 중심이 원점, 데크 윗면 z = 0, `target_position_m`은 **박스 중심**입니다(`world.add_box`, 데모 박스 z = 0.10 / 높이 0.20). 타임라인 팔레트 프레임(모서리 원점, 회전 AABB의 최소 모서리)은 X×Y 팔레트에서 원점이 (-X/2, -Y/2)이므로 `중심 = min + dims/2 - (X/2, Y/2, 0)`으로 바꿉니다. 크기는 원래 박스 크기 + `yaw_rad`(90° 회전)로 보냅니다.

```bash
# 터미널 1: 팀 모노레포 (예: ~/pac2026), 최초 1회 설치
cd ~/pac2026
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,sim]'          # pybullet, networkx, aiohttp, Pillow (README의 requirements-ahead-sim.txt는 main에 없음)
python3 scripts/run_ahead_simulator.py --no-browser     # 브라우저로 http://127.0.0.1:4173 열기

# 터미널 2: 이 저장소
cd ~/pac-mission1-lookahead
python3 tools/realtime/live_sim_bridge.py reports/realtime_test0.json --speed 1
python3 tools/realtime/live_sim_bridge.py reports/realtime_test0.json --dry-run --speed 0   # 시뮬레이터 없이 요청만 출력
```

옵션: `--speed 2`(2배속, 0 = 대기 없음), `--direct`(로봇 없이 `/api/place`), `--no-reset`, `--start`/`--limit`(일부 이벤트), `--ready-timeout`, `--url`.
동작: 이벤트 시각까지 기다림 → PLACE/RETRIEVE는 `/api/robot_place` → `/api/state`를 폴링해 로봇이 READY가 될 때까지 대기, 결과(xy/z 오차, 기울기 또는 FAILED) 출력 → `pallet_closed`이면 `/api/reset`(로봇 셀도 다시 로드). BUFFER_CURRENT와 부분 재배치는 시뮬레이터에 해당 기능이 없어 로그만 남깁니다.

주의:
- 시뮬레이터 팔레트는 1.1 × 1.1 m(`config/default.yaml`), 높이 한계는 1.5 m(pac_common을 못 찾으면 1.6 m)입니다. 타임라인 팔레트(예: 1.2 × 0.8)가 다르면 경고를 출력하고 시뮬레이터 팔레트 중앙에 맞춰 놓습니다(넘치면 overhang 경고). 같은 크기로 보려면 `config/ahead_simulator.yaml`을 복사해 `pallet:`에 `length_m`, `width_m`, `max_height_m`을 적고 `--config 복사본`으로 실행합니다.
- PyBullet 로봇은 관절 속도 25%로 움직여 박스당 시간이 타임라인(약 8 s)보다 길 수 있습니다. 이때 상태 줄에 `behind schedule`이 붙고 재생은 로봇 속도에 맞춰 진행됩니다. 빠르게 보려면 `--direct`.

## B. Gazebo 재생 노드

`GazeboReplayCore`가 이벤트를 팀 `GazeboDriverCore.on_command` 명령(`action`, `state_version`, `box_id`, `target_min_corner` [x, y, z, yaw], `robot` {q_place, q_approach})으로 바꾸고, 관절값은 6단계 `RobotFeasibility.validate_robot_motion`(`config/taehyeon/robot_check_gazebo.yaml`, 받침대 0.5 m 위 로봇, 팔레트 중심 월드 (1.35, -1.0), 데크 윗면 0.15)으로 구합니다. 6단계가 거부한 배치는 로그를 남기고 로봇 동작 없이 박스만 생성합니다. `moved`는 PARTIAL_REPACK(스냅샷 위치로 다시 생성), `pallet_closed`는 PALLET_CLOSE(팔레트 박스 삭제)입니다. 컨베이어 창(픽 위치의 현재 박스 + 상류의 보이는 박스)은 `conveyor_main` 위에 고정 박스(`conv_<id>`)로 보여 줍니다.

```bash
# 터미널 1: 팀 모노레포 ROS 워크스페이스 (서브모듈 포함 clone, colcon build 완료 상태)
source /opt/ros/humble/setup.bash
source ~/pac2026/ros2_ws/install/setup.bash
ros2 launch ~/pac2026/tools/runtime/launch/hdr50_pedestal_workcell.launch.py

# 터미널 2: 재생 (시스템 python3, rclpy 사용)
source /opt/ros/humble/setup.bash
source ~/pac2026/ros2_ws/install/setup.bash
export PAC_COMMON_SRC=~/pac2026/ros2_ws/src/pac_common \
       PAC_ROBOT_CHECK_SRC=~/pac2026/ros2_ws/src/pac_robot_check \
       PAC_RUNTIME_SRC=~/pac2026/ros2_ws/src/pac_runtime
cd ~/pac-mission1-lookahead
python3 tools/realtime/gazebo_replay.py reports/realtime_test0.json --speed 1 \
    --robot-config ~/pac2026/config/taehyeon/robot_check_gazebo.yaml
python3 tools/realtime/gazebo_replay.py reports/realtime_test0.json --dry-run    # ROS 없이 계획만 출력
```

옵션: `--motion-speed 0.3`(관절 속도 한계 대비 비율), `--no-conveyor`, `--world ahead_workcell_v2`, `--trajectory-topic /joint_trajectory_controller/joint_trajectory`, `--start`/`--limit`.
환경변수를 주지 않으면 `scripts/taehyeon/team_paths.py`(`.deps/team` 추출본 포함)에서 찾습니다. Gazebo 데크는 1.2 × 1.0 m이며 타임라인 팔레트가 다르면 중앙에 맞추고 경고합니다.

## C. HTML 뷰어

```bash
python3 tools/realtime/export_viewer.py reports/realtime_test0.json reports/realtime_test0.html
```

팔레트 스냅샷(이번 박스는 주황색), 왼쪽 앞 컨베이어 창(픽 위치 박스는 빨간 테두리), 오른쪽 버퍼 슬롯, 패널(시간, 동작, compute_s/replans/idle, fill, 버퍼), 재생/일시정지, 속도(1-100배), 이벤트 슬라이더. 마우스 드래그로 회전, 휠로 확대.

## 테스트

```bash
.venv/bin/python -m pytest -q tests/taehyeon/test_th_realtime_demo.py
```

ROS·Gazebo·PyBullet·네트워크 없이 순수 로직만 검사합니다(좌표 변환, payload, 팔레트 경고, 가짜 시뮬레이터로 브리지 재생, 6단계 관절값, 재배치·팔레트 교체, HTML).

## 한계 (여기서 검증하지 못한 것)

- 실제 시뮬레이터/Gazebo에 연결해 돌려 보지 못했습니다(이 환경에 pybullet, aiohttp, ROS, Gazebo 없음). HTTP API·좌표는 코드 읽기와 `--dry-run`, 가짜 서버 테스트로만 확인했습니다. PyBullet 로봇 셀이 1.2 m 팔레트 가장자리까지 닿는지도 미확인입니다.
- `pallet_closed`인 단계에서 놓은 박스는 타임라인에 목표 위치가 없습니다(`TimedRun`이 팔레트 교체로 비워진 뒤 `target`을 계산). 두 도구 모두 이 박스는 건너뛰고 로그를 남깁니다. `realtime.py`에서 `w.step()` 전후 비교를 교체 전에 하도록 고치면 해결됩니다.
- 버퍼 박스와 부분 재배치는 Live 시뮬레이터에 표시되지 않습니다(해당 API 없음). Gazebo에서도 버퍼 랙에는 표시하지 않습니다.
- Gazebo 재생은 흡착 없이 궤적 재생 후 목표 위치에 박스를 생성합니다(팀 `gazebo_driver`와 같은 방식). 표본 타임라인 80박스 중 17개는 6단계가 `ROBOT_COLLISION`으로 거부해 동작 없이 생성됩니다(상위 계획기가 6단계를 고려하지 않음).
- HTML 뷰어는 로컬 three.js r128로 헤드리스 Chromium에서 렌더링을 확인했습니다. cdnjs 접속은 이 환경에서 막혀 있어 확인하지 못했습니다.
