import robotic as ry

def gripper_open(C, bot: ry.BotOp, widht = 0.75, speed = 0.2):
    bot.gripperMove(ry._left, widht, speed)

    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

def gripper_close(C, bot, widht = 0.01, speed = 0.2):
    bot.gripperMove(ry._left, widht, speed)

    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

def gripper_close_grasp(C, bot, target_obj = '', force=5.00, widht = 0.01, speed = 0.2):
    bot.gripperCloseGrasp(ry._left, target_obj, force, widht, speed)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)