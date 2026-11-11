from botlab import (
    Model,
    PolicyController,
    concat,
    constant,
    mode_select,
    phase_clock,
    policy_bank,
    state_machine,
    where,
)
from microduck_policy import (
    GROUND_PICK,
    GROUND_PICK_PERIOD_STEPS,
    KICK_LEFT,
    KICK_RIGHT,
    MODEL_SPECS,
    POLICY_GROUND_PICK,
    POLICY_KICK_LEFT,
    POLICY_KICK_RIGHT,
    POLICY_ROLL,
    POLICY_SITSTAND,
    POLICY_STAND,
    POLICY_WALK,
    POST_KICK,
    RECOVER,
    RECOVER_ENTER,
    RECOVER_UPRIGHT,
    ROLL,
    ROLL_TIPPED,
    SIT,
    SIT_ENTER,
    SIT_EXIT,
    WALK,
    WALK_RESET,
    policies,
    transitions,
)


class MicroDuck:
    """Project-local MicroDuck library, analogous to an Arduino device library."""

    MODEL_ROOT = "assets/models"

    @staticmethod
    def parameters():
        return {
            "forward_velocity": {
                "label": "Forward speed",
                "default": 0,
                "range": [-0.2, 0.25],
                "step": 0.05,
                "unit": "m/s",
            },
            "lateral_velocity": {
                "label": "Lateral speed",
                "default": 0,
                "range": [0, 0],
                "step": 0.05,
                "unit": "m/s",
            },
            "yaw_rate": {
                "label": "Yaw rate",
                "default": 0,
                "range": [-1, 1],
                "step": 0.1,
                "unit": "rad/s",
            },
            "policy_command": {
                "label": "Policy command",
                "default": 0,
                "range": [0, 6],
                "step": 1,
            },
        }

    def __init__(self, robot, model_root=MODEL_ROOT):
        self.robot = robot
        self.models = {
            key: Model(
                f"{model_root}/{filename}",
                provider="wasm",
                name=name,
            )
            for key, _mode, filename, name in MODEL_SPECS
        }

    def _velocity_command(self):
        return concat([
            self.robot.commands(
                "forward_velocity",
                "lateral_velocity",
                "yaw_rate",
                input_mode="keyboard-gamepad",
                bindings={
                    "forward_velocity": {"increase": "KeyW", "decrease": "KeyS"},
                    "yaw_rate": {"increase": "KeyA", "decrease": "KeyD"},
                },
            ),
            constant(0, size=10),
        ])

    def _policy_command(self):
        return self.robot.commands(
            "policy_command",
            mappings=[
                {"key": "KeyX", "value": 1, "label": "Sit / stand"},
                {"key": "KeyZ", "value": 2, "label": "Kick left"},
                {"key": "KeyC", "value": 3, "label": "Kick right"},
                {"key": "KeyV", "value": 4, "label": "Roll"},
                {"key": "KeyB", "value": 5, "label": "Ground pick"},
                {"key": "KeyR", "value": 6, "label": "Stand recovery"},
            ],
        )

    def _state_machine(self, policy_command, gravity):
        return state_machine(
            policy_command,
            initial=WALK,
            transitions=transitions(
                tipped=gravity.at(2) > -0.3,
                upright=gravity.at(2) < -0.85,
            ),
        )

    def _command(self, machine, velocity_command):
        zero_command = constant(0, size=13)
        sit_command = concat([constant(1, size=1), constant(0, size=12)])
        ground_pick_command = concat([
            phase_clock(machine.steps, period_steps=GROUND_PICK_PERIOD_STEPS),
            constant(0, size=11),
        ])
        return mode_select(
            machine.state,
            zero_command,
            choices={
                (WALK, WALK_RESET): velocity_command,
                SIT: sit_command,
                GROUND_PICK: ground_pick_command,
            },
            passive=-1,
        )

    def _observation(self, machine, angular_velocity, gravity, command):
        clear_last_action = (
            ((machine.state == SIT_ENTER) & (machine.steps == 0))
            | (machine.state == WALK_RESET)
            | (machine.state == RECOVER_ENTER)
        )
        last_action = where(
            clear_last_action,
            constant(0, size=14),
            self.robot.last_action(),
        )
        return concat([
            angular_velocity,
            gravity,
            self.robot.joint_position(default=self.robot.default_angles),
            self.robot.joint_velocity(),
            last_action,
            command,
        ])

    @staticmethod
    def _policy_mode(state):
        return mode_select(
            state,
            constant(POLICY_WALK, size=1),
            choices={
                (WALK, POST_KICK, WALK_RESET): constant(POLICY_WALK, size=1),
                (SIT_ENTER, SIT, SIT_EXIT): constant(POLICY_SITSTAND, size=1),
                KICK_LEFT: constant(POLICY_KICK_LEFT, size=1),
                KICK_RIGHT: constant(POLICY_KICK_RIGHT, size=1),
                (ROLL, ROLL_TIPPED): constant(POLICY_ROLL, size=1),
                GROUND_PICK: constant(POLICY_GROUND_PICK, size=1),
                (RECOVER_ENTER, RECOVER, RECOVER_UPRIGHT): constant(
                    POLICY_STAND,
                    size=1,
                ),
            },
            passive=-1,
        )

    def _action(self, machine, observation):
        policy_specs = policies(self.models)
        return policy_bank(
            self._policy_mode(machine.state),
            inputs={spec["key"]: observation for spec in policy_specs},
            policies=policy_specs,
        ).action()

    def controller(self):
        velocity_command = self._velocity_command()
        # The policy reads the IMU gyro sensor (its own site frame), not qvel, so
        # base_angular_velocity() is not equivalent; sensor_data is the named-sensor primitive.
        angular_velocity = self.robot.sensor_data("imu_ang_vel", size=3)
        gravity = self.robot.projected_gravity(target="trunk_base")
        machine = self._state_machine(self._policy_command(), gravity)
        command = self._command(machine, velocity_command)
        observation = self._observation(machine, angular_velocity, gravity, command)
        action = self._action(machine, observation)

        # The MJCF position actuators consume policy position targets directly.
        control = self.robot.default_angles + self.robot.action_scale * action[
            self.robot.motor_order
        ]
        return PolicyController(self.robot, control=control)
