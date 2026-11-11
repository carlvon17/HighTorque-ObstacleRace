"""Shared visual wood-crate boards; the physical envelope remains one solid box.

Relative positions use the center of the obstacle as origin. Board construction
and yaw are visual interpretations of the photograph, not measured joinery.
"""


def crate_boards(dimensions, rgb):
    w, d, h = dimensions
    t = .012
    boards = []

    def add(name, pos, size, shade):
        boards.append({"name": name, "pos": pos, "size": size,
                       "rgb": [min(255, round(c*shade)) for c in rgb]})

    for side in (-1, 1):
        add(f"cap_{side}", (0, 0, side*(h-t)/2), (w, d, t), 1.035)
        add(f"panel_{side}", (0, side*(d-t)/2, 0), (w-2*t, t, h-2*t), .985)
        inner_d = d-2*t
        slat_width = inner_d*.29
        for i, y in enumerate((-(inner_d-slat_width)/2, 0, (inner_d-slat_width)/2)):
            add(f"slat_{side}_{i}", (side*(w-t)/2, y, 0),
                (t, slat_width, h-2*t), (.97, 1.0, 1.02)[i])
    return boards
