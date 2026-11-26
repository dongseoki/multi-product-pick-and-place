# 다품종 Pick and Place

## 개발 환경

저장소 루트에서 [uv](https://docs.astral.sh/uv/)로 실행한다. Python 3.12.16은 `.python-version`에, MuJoCo 3.3.7과 개발 의존성은 `pyproject.toml`과 `uv.lock`에 기록한다.

```bash
uv sync --locked
uv run --locked python -c "import sys, mujoco; print(sys.version.split()[0], mujoco.__version__)"
uv run --locked python -m pytest -q
```

현재 pytest는 회차 판정 로직을 검증한다. 로봇·그리퍼 모델 로드와 병목 파지 검증은 별도 작업이며, [4일차 체크리스트](docs/4일차%20장면과%20시스템%20구조%20체크리스트.md)에서 진행 상태를 확인한다.
.
