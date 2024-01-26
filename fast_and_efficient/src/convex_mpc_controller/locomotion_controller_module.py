"""A model based controller framework."""

import enum
import ml_collections
import numpy as np
from typing import Tuple, Callable

from fast_and_efficient.src.convex_mpc_controller.com_velocity_estimator import (
  COMVelocityEstimator)
from fast_and_efficient.src.convex_mpc_controller.offset_gait_generator import (
  OffsetGaitGenerator)
from fast_and_efficient.src.convex_mpc_controller.raibert_swing_leg_controller import (
  RaibertSwingLegController)
from fast_and_efficient.src.convex_mpc_controller.torque_stance_leg_controller_mpc import (
  TorqueStanceLegController)
from fast_and_efficient.src.convex_mpc_controller import (
  torque_stance_leg_controller_mpc)
from fast_and_efficient.src.convex_mpc_controller.gait_configs import (
  crawl, trot, flytrot)

from fast_and_efficient.src.robots.a1 import A1
from fast_and_efficient.src.robots.motors import MotorCommand


class ControllerMode(enum.Enum):
  DOWN = 1
  STAND = 2
  WALK = 3
  TERMINATE = 4


class GaitType(enum.Enum):
  CRAWL = 1
  TROT = 2
  FLYTROT = 3


def get_sim_conf():
  config = ml_collections.ConfigDict()
  config.timestep: float = 0.002
  config.action_repeat: int = 1
  config.reset_time_s: float = 3.
  config.num_solver_iterations: int = 30
  config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
  config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
  config.on_rack: bool = False
  return config


class LocomotionController(object):
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      robot: A1,
      gait_generator: OffsetGaitGenerator,
      state_estimator: COMVelocityEstimator,
      swing_leg_controller: RaibertSwingLegController,
      stance_leg_controller: TorqueStanceLegController,
      clock: Callable[[], float],
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      swing_leg_controller: Generates motor actions for swing legs.
      stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._robot = robot
    self._clock = clock
    self._reset_time = self._clock()
    self._time_since_reset = 0.0
    self._gait_generator = gait_generator
    self._state_estimator = state_estimator
    self._swing_leg_controller = swing_leg_controller
    self._stance_leg_controller = stance_leg_controller

    self._mode = ControllerMode.DOWN
    self.set_controller_mode(ControllerMode.STAND)
    self._gait = None
    self._desired_gait = GaitType.CRAWL
    self._handle_gait_switch()

  @property
  def swing_leg_controller(self) -> RaibertSwingLegController:
    return self._swing_leg_controller

  @property
  def stance_leg_controller(self) -> TorqueStanceLegController:
    return self._stance_leg_controller

  @property
  def gait_generator(self) -> OffsetGaitGenerator:
    return self._gait_generator

  @property
  def state_estimator(self) -> COMVelocityEstimator:
    return self._state_estimator
  
  @property
  def time_since_reset(self) -> float:
    return self._time_since_reset

  def reset_robot(self) -> None:
    self._robot.reset(hard_reset=False)

  def reset(self) -> None:
    self._reset_time = self._clock()
    self._time_since_reset = 0.0
    self._gait_generator.reset()
    self._state_estimator.reset(self._time_since_reset)
    self._swing_leg_controller.reset(self._time_since_reset)
    self._stance_leg_controller.reset(self._time_since_reset)

  def update(self) -> None:
    self._time_since_reset = self._clock() - self._reset_time
    self._gait_generator.update()
    self._state_estimator.update(self._gait_generator.desired_leg_state)
    self._swing_leg_controller.update(self._time_since_reset)
    future_contact_estimate = self._gait_generator.get_estimated_contact_states(
        torque_stance_leg_controller_mpc.PLANNING_HORIZON_STEPS,
        torque_stance_leg_controller_mpc.PLANNING_TIMESTEP)
    self._stance_leg_controller.update(self._time_since_reset,
                                       future_contact_estimate)

  def get_action(self) -> Tuple[MotorCommand, dict]:
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    swing_action = self._swing_leg_controller.get_action()
    stance_action, qp_sol = self._stance_leg_controller.get_action()

    actions = []
    for joint_id in range(self._robot.num_motors):
      if joint_id in swing_action:
        actions.append(swing_action[joint_id])
      else:
        assert joint_id in stance_action
        actions.append(stance_action[joint_id])

    vectorized_action = MotorCommand(
        desired_position=[action.desired_position for action in actions],
        kp=[action.kp for action in actions],
        desired_velocity=[action.desired_velocity for action in actions],
        kd=[action.kd for action in actions],
        desired_extra_torque=[
            action.desired_extra_torque for action in actions
        ])

    return vectorized_action, dict(qp_sol=qp_sol)

  def _get_stand_action(self) -> MotorCommand:
    return MotorCommand(
        desired_position=self._robot.motor_group.init_positions,
        kp=self._robot.motor_group.kps,
        desired_velocity=0,
        kd=self._robot.motor_group.kds,
        desired_extra_torque=0)

  def _handle_mode_switch(self) -> None:
    if self._mode == self._desired_mode:
      return
    self._mode = self._desired_mode
    if self._desired_mode == ControllerMode.DOWN:
      pass
      # logging.info("Entering joint damping mode.")
      # self._flush_logging()
    elif self._desired_mode == ControllerMode.STAND:
      # logging.info("Standing up.")
      self.reset_robot()
    else:
      # logging.info("Walking.")
      self.reset()

  def _handle_gait_switch(self) -> None:
    if self._gait == self._desired_gait:
      return
    if self._desired_gait == GaitType.CRAWL:
      # logging.info("Switched to Crawling gait.")
      self._gait_config = crawl.get_config()
    elif self._desired_gait == GaitType.TROT:
      # logging.info("Switched  to Trotting gait.")
      self._gait_config = trot.get_config()
    else:
      # logging.info("Switched to Fly-Trotting gait.")
      self._gait_config = flytrot.get_config()

    self._gait = self._desired_gait
    self._gait_generator.gait_params = self._gait_config.gait_parameters
    self._swing_leg_controller.foot_height = self._gait_config.foot_clearance_max
    self._swing_leg_controller.foot_landing_clearance = \
      self._gait_config.foot_clearance_land
    
  def set_controller_mode(self, mode: ControllerMode) -> None:
    self._desired_mode = mode

  def set_gait(self, gait: GaitType) -> None:
    self._desired_gait = gait

  @property
  def is_safe(self) -> bool:
    if self.mode != ControllerMode.WALK:
      return True
    rot_mat = np.array(
        self._robot.pybullet_client.getMatrixFromQuaternion(
            self._state_estimator.com_orientation_quat_ground_frame)).reshape(
                (3, 3))
    up_vec = rot_mat[2, 2]
    base_height = self._robot.base_position[2]
    return up_vec > 0.85 and base_height > 0.18

  @property
  def mode(self) -> ControllerMode:
    return self._mode

  def set_desired_speed(
    self,
    desired_lin_speed_ratio: Tuple[float, float],
    desired_rot_speed_ratio: float
  ) -> None:
    desired_lin_speed = (
        self._gait_config.max_forward_speed * desired_lin_speed_ratio[0],
        self._gait_config.max_side_speed * desired_lin_speed_ratio[1],
        0,
    )
    desired_rot_speed = \
      self._gait_config.max_rot_speed * desired_rot_speed_ratio
    self._swing_leg_controller.desired_speed = desired_lin_speed
    self._swing_leg_controller.desired_twisting_speed = desired_rot_speed
    self._stance_leg_controller.desired_speed = desired_lin_speed
    self._stance_leg_controller.desired_twisting_speed = desired_rot_speed


  def set_gait_parameters(self, gait_parameters):
    raise NotImplementedError()

  def set_qp_weight(self, qp_weight):
    raise NotImplementedError()

  def set_mpc_mass(self, mpc_mass):
    raise NotImplementedError()

  def set_mpc_inertia(self, mpc_inertia):
    raise NotImplementedError()

  def set_mpc_foot_friction(self, mpc_foot_friction):
    raise NotImplementedError()

  def set_foot_landing_clearance(self, foot_landing_clearance):
    raise NotImplementedError()

