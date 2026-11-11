"""Continuous head motion profile inside the verified mechanical work range."""
import math


def bounded_absolute_target(current, target, lower, upper):
    if not all(math.isfinite(v) for v in (current, target, lower, upper)):
        raise ValueError('Non-finite head command')
    lo, hi = max(-1.4, lower+.1), min(1.4, upper-.1)
    if not lo <= current <= hi or not lo <= target <= hi:
        raise ValueError('Head target outside bounded work range')
    return float(target)


def smooth_setpoint(start, target, elapsed, duration):
    if not all(math.isfinite(v) for v in (start, target, elapsed, duration)) or duration <= 0:
        raise ValueError('Invalid head motion profile')
    u = min(1., max(0., elapsed/duration))
    return start+(target-start)*(u*u*(3.-2.*u))


def validate_body_only_amp(config, body_names):
    names=config.get('joint_names',[])
    if (config.get('algorithm')!='amp' or config.get('motor_group')!='all-amp'
            or config.get('dofs')!=20 or len(names)!=20 or len(set(names))!=20
            or set(names)!=set(body_names)
            or any('head' in name for name in names)):
        raise ValueError('Running policy must control exactly the 20 body joints and exclude head')
    policy_file=config.get('policy_name')
    if not isinstance(policy_file,str) or not policy_file:
        raise ValueError('Running policy filename is unavailable')
    return policy_file
