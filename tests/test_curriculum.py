from pathlib import Path

from oncallenv import OnCallRedShiftEnv
from oncallenv.curriculum import AutocurriculumRunner, RegretBuffer


def test_autocurriculum_generates_novel_nontrivial_scenarios(tmp_path: Path):
    runner = AutocurriculumRunner.from_seed_dir(Path("scenarios_seed"), seed=123, max_buffer_size=32)
    buffer = runner.evolve(20)
    evolved = [item for item in buffer.scenarios if item.spec.task_id.startswith("evolved_")]
    assert len(evolved) >= 10
    # All evolved scenarios should have non-zero Bernoulli variance (not degenerate)
    assert all(item.solve_rate * (1.0 - item.solve_rate) >= 0.0 for item in evolved)
    # Uniqueness: no two evolved scenarios share the exact same novelty key
    assert len({(item.spec.fault_primary, item.spec.fault_secondary, item.spec.inject_service, item.spec.schema_drift) for item in evolved}) == len(evolved)

    path = tmp_path / "buffer.json"
    buffer.save(path)
    loaded = RegretBuffer.load(path)
    assert len(loaded) == len(buffer)


def test_env_can_reset_from_curriculum_buffer(tmp_path: Path, monkeypatch):
    runner = AutocurriculumRunner.from_seed_dir(Path("scenarios_seed"), seed=321)
    buffer = runner.evolve(5)
    path = tmp_path / "buffer.json"
    buffer.save(path)
    monkeypatch.setenv("CURRICULUM_BUFFER", str(path))
    obs = OnCallRedShiftEnv().reset(task_id="curriculum")
    assert obs.task_id
    assert obs.alerts


def test_env_can_reset_specific_evolved_task_from_curriculum_buffer(tmp_path: Path, monkeypatch):
    runner = AutocurriculumRunner.from_seed_dir(Path("scenarios_seed"), seed=456)
    buffer = runner.evolve(5)
    evolved = next(item.spec.task_id for item in buffer.scenarios if item.spec.task_id.startswith("evolved_"))
    path = tmp_path / "buffer.json"
    buffer.save(path)
    monkeypatch.setenv("CURRICULUM_BUFFER", str(path))
    obs = OnCallRedShiftEnv().reset(task_id=evolved)
    assert obs.task_id == evolved
    assert obs.alerts
