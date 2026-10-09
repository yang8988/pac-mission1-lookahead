# PAC Mission 1 – Look-ahead 적재 알고리즘 (태현)

컨베이어 위에 보이는 **다음 N개 박스**를 이용해, 학습 없이 실시간 탐색으로 4단계 결정(지금 박스 놓기 / 버퍼에 두기 / 버퍼 박스 꺼내기)을 내리는 적재 규칙입니다.
기존 팀 리포지토리 [`yang8988/pac-mission1-shared`](https://github.com/yang8988/pac-mission1-shared) 의 태현 파트(4단계, 5-① 후보 생성, 5-② Hard Mask)를 바탕으로 합니다.
팀원 코드는 이 리포지토리에 복사하지 않고, 실행할 때 팀 리포지토리에서 읽기 전용으로 가져옵니다.

자세한 규칙·점수식·결과: [`docs/lookahead.md`](docs/lookahead.md)

## 구성

| 경로 | 내용 |
|---|---|
| `ros2_ws/src/pac_highlevel/pac_highlevel/lookahead.py` | **N개 탐색 정책** (`LookaheadPolicy`) |
| `ros2_ws/src/pac_highlevel/` | 4단계 시뮬레이터, Rule 정책, 버퍼·마감·재적재 규칙 |
| `ros2_ws/src/pac_candidates/` | 5-① 후보 생성(EMS+EP), 5-② Hard Mask(지지·하중·높이) |
| `tools/virtual_data/` | 생성기 데이터 → 박스 흐름 변환, 가상 셀 |
| `config/taehyeon/lookahead.yaml` | 탐색 설정 (N, 모드, 시간 제한, 점수 가중치) |
| `tools/highlevel/scripts/evaluate_lookahead.py` | Rule과 같은 박스 흐름으로 짝지어 비교 |
| `tools/highlevel/scripts/make_dataset.py` | 평가 데이터셋 생성 (재성 님 생성기) |
| `scripts/taehyeon/fetch_team_deps.sh` | 팀원 패키지(pac_common, 생성기)를 `.deps/`로 가져옴 |
| `tests/taehyeon/` | 4단계·탐색 테스트 |

## 실행 (WSL / Linux, Python 3.10+)

```bash
git clone https://github.com/yang8988/pac-mission1-lookahead.git
cd pac-mission1-lookahead
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyyaml pytest

bash scripts/taehyeon/fetch_team_deps.sh          # 팀원 패키지 (읽기 전용)
python -m pytest -q tests/taehyeon                # 테스트
python tools/highlevel/scripts/make_dataset.py    # 평가 데이터 240개 시나리오 (몇 분)

# Rule vs N개 탐색 (검증 36개 시나리오, 4코어 약 30분)
python tools/highlevel/scripts/evaluate_lookahead.py \
    --dataset tools/highlevel/output/dataset80_x40_s8 --split val \
    --variant N3=horizon=3 --output reports/lookahead_val.json
```

`--variant 이름=키=값,키=값` 으로 `config/taehyeon/lookahead.yaml` 의 항목을 바꿔 여러 설정을 한 번에 비교할 수 있습니다 (예: `N5=horizon=5,mode=beam`).

## 코드에서 쓰기

```python
from pac_highlevel import LookaheadPolicy, load_highlevel_config, load_lookahead_config, run_policy

hl = load_highlevel_config("config/taehyeon/highlevel.yaml")
policy = LookaheadPolicy(hl, load_lookahead_config("config/taehyeon/lookahead.yaml"))
action = policy(world)          # world: PalletizingWorld, world.arrivals 에 보이는 박스까지 들어 있음
```

탐색은 `world.arrivals[world.next_arrival : world.next_arrival + horizon]` 까지만 봅니다. 실제 셀에서는 컨베이어 카메라가 본 박스를 이 목록에 넣으면 됩니다.
