"""시험 관측 기록에 D03 합격 기준을 적용한다."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialResult:
    """시험 한 회차에서 수집한 관측 결과를 담는다.

    Attributes:
        print_contact: 로봇팔 또는 그리퍼가 대상 병의 몸통에 닿았는지 여부.
        dropped: 들어 올린 병이 내려놓기 전에 떨어졌는지 여부.
        seated: 병을 놓은 뒤 1초 동안 병이 자기 바닥면으로 목표 칸 바닥에
            지지되고, 바닥면 전체가 칸 안에 머물렀는지 여부.
        neighboring_bottle_contact: 로봇팔, 그리퍼 또는 대상 병이 이미 적재된
            병에 닿았는지 여부.
        elapsed_s: 자동 동작 시작부터 대기 자세 복귀까지의 시뮬레이션 시간(초).
    """

    print_contact: bool
    dropped: bool
    seated: bool
    neighboring_bottle_contact: bool
    elapsed_s: float


def judge_episode(trials: Sequence[TrialResult]) -> bool:
    """10회 시험이 D03의 필수·안전 기준을 충족하는지 판정한다.

    Args:
        trials: 회차별 관측 결과.

    Returns:
        관측 결과가 정확히 10개이고, 목표 칸 안착이 9회 이상이며,
        인쇄면 접촉·낙하·기존 병 접촉이 모두 0회이면 True를 반환한다.
        그 외에는 False를 반환한다.
    """
    # R-01~R-04: 사전에 정한 10개 좌표의 회차 결과가 모두 필요하다.
    if len(trials) != 10:
        return False

    # R-01 / T-01, R-02 / T-02, R-04 / T-04: 금지 사건 한 번이면 불합격이다.
    prohibited_event = any(
        trial.print_contact or trial.dropped or trial.neighboring_bottle_contact
        for trial in trials
    )
    # R-03 / T-03: 목표 칸 안착은 10회 중 9회 이상이어야 한다.
    return not prohibited_event and sum(trial.seated for trial in trials) >= 9


def judge_cycle_time(trial: TrialResult, limit_s: float | None = None) -> bool | None:
    """한 회차의 사이클타임이 목표 상한을 충족하는지 판정한다.

    Args:
        trial: 판정할 회차의 관측 결과.
        limit_s: 사이클타임 상한(초). 상한이 미정이면 None을 사용한다.

    Returns:
        회차 시간이 상한 이하면 True, 초과하면 False를 반환한다.
        상한이 미정이면 None을 반환한다.
    """
    # R-05 / T-05: 시간 목표는 별도로 판정하며 상한과 같은 값도 달성이다.
    if limit_s is None:
        return None
    return trial.elapsed_s <= limit_s
