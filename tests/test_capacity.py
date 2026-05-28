from src.rl_traffic.capacity import (
    compute_directional_values,
    compute_wasted_green,
)


def test_no_event_effective_values_match_raw_totals():
    queue = {"N": 10.0, "E": 20.0, "S": 15.0, "W": 5.0}
    wait = {"N": 1.0, "E": 2.0, "S": 3.0, "W": 4.0}
    tail = {"N": 2.0, "E": 3.0, "S": 4.0, "W": 5.0}
    capacity = {"N": 1.0, "E": 1.0, "S": 1.0, "W": 1.0}

    values = compute_directional_values(queue, wait, tail, capacity)

    assert values.total_queue == 50.0
    assert values.effective_queue == 50.0
    assert values.total_wait == 10.0
    assert values.effective_wait == 10.0
    assert compute_wasted_green(["N", "S"], capacity) == 0.0


def test_full_block_excludes_blocked_direction_from_effective_queue():
    queue = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    wait = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    tail = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    capacity = {"N": 0.0, "E": 1.0, "S": 1.0, "W": 1.0}

    values = compute_directional_values(queue, wait, tail, capacity)

    assert values.total_queue == 145.0
    assert values.effective_queue == 45.0
    assert compute_wasted_green(["N"], capacity) == 1.0
    assert compute_wasted_green(["N", "S"], capacity) == 0.5
    assert compute_wasted_green(["E", "W"], capacity) == 0.0


def test_partial_block_scales_effective_values_without_wasted_green():
    queue = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    wait = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    tail = {"N": 100.0, "E": 20.0, "S": 15.0, "W": 10.0}
    capacity = {"N": 0.5, "E": 1.0, "S": 1.0, "W": 1.0}

    values = compute_directional_values(queue, wait, tail, capacity)

    assert values.total_queue == 145.0
    assert values.effective_queue == 95.0
    assert compute_wasted_green(["N"], capacity) == 0.0
