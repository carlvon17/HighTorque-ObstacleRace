from botlab import Robot, project
from microduck import MicroDuck


# Arduino-style sketch: keep the project and hardware visible here, while the
# project-local MicroDuck library owns its observation and policy state machine.
project(
    "MicroDuck Athletics",
    control_hz=50,
    physics_hz=200,
    parameters=MicroDuck.parameters(),
)

robot = Robot(
    "assets/robot/scene.xml",
    preset="microduck.preset.yaml",
    name="MicroDuck",
)
duck = MicroDuck(robot)


def create_controller(sim):
    return duck.controller()
