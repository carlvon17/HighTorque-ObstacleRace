# Project-local MicroDuck policy orchestration.
# These state numbers and timings reproduce the browser runtime; they are not SDK APIs.

WALK = 0
SIT_ENTER = 1
SIT = 2
SIT_EXIT = 3
KICK_LEFT = 4
KICK_RIGHT = 5
POST_KICK = 6
ROLL = 7
ROLL_TIPPED = 8
GROUND_PICK = 9
WALK_RESET = 10
RECOVER_ENTER = 11
RECOVER = 12
RECOVER_UPRIGHT = 13

POLICY_WALK = 0
POLICY_SITSTAND = 1
POLICY_KICK_LEFT = 2
POLICY_KICK_RIGHT = 3
POLICY_ROLL = 4
POLICY_GROUND_PICK = 5
POLICY_STAND = 6

MODEL_SPECS = (
    ("walk", POLICY_WALK, "walking.onnx", "Walking Policy"),
    ("sitstand", POLICY_SITSTAND, "sitstand.onnx", "Sit/Stand Policy"),
    ("kick_left", POLICY_KICK_LEFT, "kick-left.onnx", "Left Kick Policy"),
    ("kick_right", POLICY_KICK_RIGHT, "kick-right.onnx", "Right Kick Policy"),
    ("roll", POLICY_ROLL, "roll.onnx", "Roll Policy"),
    ("ground_pick", POLICY_GROUND_PICK, "ground-pick.onnx", "Ground Pick Policy"),
    ("stand", POLICY_STAND, "stand.onnx", "Stand Recovery Policy"),
)

COMMAND_SIT_TOGGLE = 1
COMMAND_KICK_LEFT = 2
COMMAND_KICK_RIGHT = 3
COMMAND_ROLL = 4
COMMAND_GROUND_PICK = 5
COMMAND_RECOVER = 6

SIT_ENTER_STEPS = 40       # 0.8 s
SIT_EXIT_STEPS = 100       # 2.0 s
KICK_STEPS = 25            # 0.5 s
POST_KICK_STEPS = 20       # 0.4 s
ROLL_MIN_STEPS = 40
ROLL_TIMEOUT_STEPS = 150   # 3.0 s
GROUND_PICK_PERIOD_STEPS = 200
GROUND_PICK_STEPS = 140    # phase 0.7 of a 4 s period
RECOVER_UPRIGHT_STEPS = 50  # 1.0 s


def transitions(tipped, upright):
    return [
        {"from": WALK, "command": COMMAND_SIT_TOGGLE, "to": SIT_ENTER},
        {"from": SIT_ENTER, "after_steps": SIT_ENTER_STEPS, "to": SIT},
        {"from": SIT, "command": COMMAND_SIT_TOGGLE, "to": SIT_EXIT},
        {"from": SIT_EXIT, "after_steps": SIT_EXIT_STEPS, "to": WALK_RESET},
        {"from": WALK, "command": COMMAND_KICK_LEFT, "to": KICK_LEFT},
        {"from": WALK, "command": COMMAND_KICK_RIGHT, "to": KICK_RIGHT},
        {"from": KICK_LEFT, "after_steps": KICK_STEPS, "to": POST_KICK},
        {"from": KICK_RIGHT, "after_steps": KICK_STEPS, "to": POST_KICK},
        {"from": POST_KICK, "after_steps": POST_KICK_STEPS, "to": WALK},
        {"from": WALK, "command": COMMAND_ROLL, "to": ROLL},
        {
            "from": ROLL,
            "when": tipped,
            "to": ROLL_TIPPED,
            "reset_steps": False,
        },
        {"from": ROLL, "after_steps": ROLL_TIMEOUT_STEPS, "to": WALK_RESET},
        {
            "from": ROLL_TIPPED,
            "after_steps": ROLL_MIN_STEPS,
            "when": upright,
            "to": WALK_RESET,
        },
        {
            "from": ROLL_TIPPED,
            "after_steps": ROLL_TIMEOUT_STEPS,
            "to": WALK_RESET,
        },
        {"from": WALK, "command": COMMAND_GROUND_PICK, "to": GROUND_PICK},
        {"from": GROUND_PICK, "after_steps": GROUND_PICK_STEPS, "to": WALK},
        {"from": WALK, "command": COMMAND_RECOVER, "to": RECOVER_ENTER},
        {"from": RECOVER_ENTER, "after_steps": 1, "to": RECOVER},
        {"from": RECOVER, "when": upright, "to": RECOVER_UPRIGHT},
        {"from": RECOVER_UPRIGHT, "when": ~upright, "to": RECOVER},
        {
            "from": RECOVER_UPRIGHT,
            "after_steps": RECOVER_UPRIGHT_STEPS,
            "to": WALK_RESET,
        },
        {"from": WALK_RESET, "after_steps": 1, "to": WALK},
    ]


def policies(models):
    return [
        {
            "key": key,
            "mode": mode,
            "model": models[key],
            "input_name": "obs",
            "output_name": "actions",
        }
        for key, mode, _filename, _name in MODEL_SPECS
    ]
