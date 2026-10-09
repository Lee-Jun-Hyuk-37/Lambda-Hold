"""Low-level helpers to reach the underlying SCONE gaitgym env and its model."""


def get_inner_gaitgym(env):
    """Walk down the wrapper chain to the underlying gaitgym env."""
    inner = env
    while hasattr(inner, "_env") or hasattr(inner, "env"):
        inner = inner._env if hasattr(inner, "_env") else inner.env
    return inner
