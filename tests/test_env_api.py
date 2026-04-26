from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action, Observation, State


def test_gym_style_api_contract():
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id="seed_easy_memory_leak")
    assert isinstance(obs, Observation)
    assert "kubectl_logs" in obs.available_tools
    obs = env.step(Action(command="kubectl_get_pods"))
    assert isinstance(obs, Observation)
    state = env.state
    assert isinstance(state, State)
    assert state.step_count == 1
    env.close()
    assert env.state.done is True

