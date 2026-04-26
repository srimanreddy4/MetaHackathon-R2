"""FastAPI application for the OnCallEnv Red Shift environment.

Exposes the environment over HTTP and WebSocket endpoints compatible with
the standard OpenEnv EnvClient protocol.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core>=0.2.0"
    ) from e

try:
    from ..models import OnCallRedShiftAction, OnCallRedShiftObservation
    from .oncallenv_redshift_environment import OnCallRedShiftEnvironment
except ImportError:
    from models import OnCallRedShiftAction, OnCallRedShiftObservation
    from server.oncallenv_redshift_environment import OnCallRedShiftEnvironment

app = create_app(
    OnCallRedShiftEnvironment,
    OnCallRedShiftAction,
    OnCallRedShiftObservation,
    env_name="oncallenv_redshift",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
