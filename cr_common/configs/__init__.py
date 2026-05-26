"""State estimation configuration utilities."""


def merge_configs(robot_cfg: dict, est_cfg: dict) -> dict:
    """Shallow-merge estimation config on top of robot config.

    For keys present in both, if both values are dicts they are merged
    (estimation overrides robot). Otherwise the estimation value wins.
    This lets the estimation yaml override e.g. ``rod.proximal_node_idx``
    while inheriting the rest from the robot yaml.
    """
    merged = {}
    for key in set(list(robot_cfg.keys()) + list(est_cfg.keys())):
        if key in est_cfg and key in robot_cfg:
            if isinstance(est_cfg[key], dict) and isinstance(robot_cfg[key], dict):
                merged[key] = {**robot_cfg[key], **est_cfg[key]}
            else:
                merged[key] = est_cfg[key]
        elif key in est_cfg:
            merged[key] = est_cfg[key]
        else:
            merged[key] = robot_cfg[key]
    return merged
