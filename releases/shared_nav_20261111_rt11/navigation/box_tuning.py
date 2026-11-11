"""Operator-facing units are centimetres and centimetres per second."""
from dataclasses import asdict, dataclass
import math

# field, label, unit, minimum, maximum, slider increment
FIELDS = (
    ('side_step', '横移一步', 'cm', 1., 9., .1),
    ('forward_step', '前进一步', 'cm', 1., 12., .1),
    ('side_speed', '横移速度上限', 'cm/s', 10., 30., 1.),
    ('forward_speed', '前进速度上限', 'cm/s', 20., 30., 1.),
    ('first_gap', '首箱横向让开距离', 'cm', 36., 45., .5),
)


@dataclass(frozen=True)
class BoxTuning:
    side_step: float = 4.6
    forward_step: float = 6.3
    side_speed: float = 26.
    forward_speed: float = 22.
    first_gap: float = 39.

    @classmethod
    def read(cls, values):
        if not isinstance(values,dict) or set(values)!=set(asdict(cls())):
            raise ValueError('请提交完整的五项设置')
        for key,label,unit,lo,hi,increment in FIELDS:
            value=values[key]
            if (type(value) not in (int,float) or not math.isfinite(value) or
                    not lo<=value<=hi or abs(round((value-lo)/increment)*increment+lo-value)>1e-6):
                raise ValueError(f'{label}允许 {lo:g}–{hi:g} {unit}，步进 {increment:g}')
        return cls(**values)
