from openenv.core.env_server.http_server import create_app

from oncallenv.core.env import OnCallRedShiftEnv
from models import Action, Observation

app = create_app(
    OnCallRedShiftEnv,
    Action,
    Observation,
    env_name="oncallenv-redshift",
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

