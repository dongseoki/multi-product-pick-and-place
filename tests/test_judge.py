"""D03 회차 집계 규칙을 가상의 관측 결과로 검증한다."""

from dataclasses import replace

import pytest

from src.judge import TrialResult, judge_cycle_time, judge_episode


SUCCESS = TrialResult(
    print_contact=False,
    dropped=False,
    seated=True,
    neighboring_bottle_contact=False,
    box_wall_contact=False,
    elapsed_s=8.4,
)


def test_nine_seated_pass():
    """T-03: 금지 사건 없이 정확히 9회 안착하면 합격한다."""
    assert judge_episode([SUCCESS] * 9 + [replace(SUCCESS, seated=False)])


def test_eight_seated_fail():
    """T-03: 금지 사건이 없어도 안착이 8회면 불합격한다."""
    assert not judge_episode([SUCCESS] * 8 + [replace(SUCCESS, seated=False)] * 2)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param("print_contact", id="T-01"),
        pytest.param("dropped", id="T-02"),
        pytest.param("neighboring_bottle_contact", id="T-04"),
    ],
)
def test_prohibited_event_fails_even_if_all_bottles_seated(event):
    """T-01·T-02·T-04: 모두 안착해도 금지 사건이 한 번이면 불합격한다."""
    assert not judge_episode([SUCCESS] * 9 + [replace(SUCCESS, **{event: True})])


@pytest.mark.parametrize("count", [0, 9, 11])
def test_requires_exactly_ten_trials(count):
    """D03 시험 조건: 관측 기록이 정확히 10개일 때만 합격할 수 있다."""
    assert not judge_episode([SUCCESS] * count)


def test_box_wall_contact_does_not_affect_episode_pass():
    """T-06 집계 확인: 벽 접촉이 기록된 회차도 합격 집계에 포함한다."""
    assert judge_episode([SUCCESS] * 9 + [replace(SUCCESS, box_wall_contact=True)])


def test_cycle_time_is_judged_separately():
    """T-05: 상한 이하는 달성이고, 초과해도 전체 합격 기준은 유지된다."""
    slow_trial = replace(SUCCESS, elapsed_s=8.5)

    assert judge_cycle_time(SUCCESS, limit_s=8.4) is True
    assert judge_cycle_time(slow_trial, limit_s=8.4) is False
    assert judge_episode([SUCCESS] * 9 + [slow_trial])


def test_cycle_time_without_limit_is_unevaluated():
    """T-05: 사이클타임 상한을 정하기 전에는 시간 목표를 판정하지 않는다."""
    assert judge_cycle_time(SUCCESS) is None
