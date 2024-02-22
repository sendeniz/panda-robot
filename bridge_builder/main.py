import robotic as ry
import numpy as np
import time
from scipy.interpolate import Rbf
from scipy.interpolate import CubicSpline
from task_env import create_n_boxes, create_construction_site, create_pillar_target_postion
from gripper_functions import gripper_open, gripper_close, gripper_close_grasp
from move import gripper_to_block, move_gripper_to, retract_gripper
print('version:', ry.__version__, ry.compiled())


C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
C = create_n_boxes(C = C, N = 3, position = 'fixed')
goal = create_construction_site(C, position = 'fixed')
pillar_goal = create_pillar_target_postion(C, position = 'fixed')
C.view()

# On real robot set True
bot = ry.BotOp(C, useRealRobot = True)
bot.home(C)

# First block
gripper_open(C, bot)
gripper_to_block(C, bot, block='box1')
retract_gripper(C, bot, distance = -0.095)
gripper_close_grasp(C, bot, target_obj = 'box1', force = 5.0,  widht = 0.045)
move_gripper_to(C, bot, add_frame_obj_str = 'pillar_target_pos1', carry_item = 'box1', orientation = 'vertical')
retract_gripper(C, bot, distance = -0.0210)
gripper_open(C, bot)
retract_gripper(C, bot, distance = 0.15)

# Second Block
gripper_open(C, bot)
gripper_to_block(C, bot, block='box2')
retract_gripper(C, bot, distance = -0.095)
gripper_close_grasp(C, bot, target_obj = 'box2', force = 5.0,  widht = 0.045)
move_gripper_to(C, bot, add_frame_obj_str = 'pillar_target_pos2', carry_item = 'box2', orientation = 'vertical')
retract_gripper(C, bot, distance = -0.0210)
gripper_open(C, bot)
retract_gripper(C, bot, distance = 0.15)

# Third Block
gripper_open(C, bot)
box_1 = C.getFrame('box1')
box_2 = C.getFrame('box2')
box_pos1 = box_1.getPosition()
box_pos2 = box_2.getPosition()
offset = 0.025
middle_between_towers = [-0.495, -0.270,  0.730]

# if real is false use below to compute middle between towers
#middle_between_towers = np.mean([box_pos1, box_pos2], 0) + [0.025, 0, offset]
C.addFrame('block3_target').setShape(ry.ST.marker, [.1, .1, .1]).setPosition(middle_between_towers).setColor([1, .0, .0])
C.view()
gripper_to_block(C, bot, block='box3')
retract_gripper(C, bot, distance = -0.095)
gripper_close_grasp(C, bot, target_obj = 'box3', force = 5.0,  widht = 0.045)
retract_gripper(C, bot, distance = 0.25)
move_gripper_to(C, bot, add_frame_obj_str = 'block3_target', carry_item = 'box3', orientation = 'horizontal')
retract_gripper(C, bot, distance = -0.065, orientation = 'horizontal')
gripper_open(C, bot)
retract_gripper(C, bot, distance = 0.05, orientation = 'horizontal')
bot.home(C)