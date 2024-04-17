import robotic as ry

def gripper_to_block(C, bot, block = 'box1', prints = False):
    qHome = C.getJointState()

    komo = ry.KOMO(C, 1, 1, 0, False)

    komo.addObjective(
        times=[], 
        feature=ry.FS.jointState, 
        frames=[],
        type=ry.OT.sos, 
        scale=[1e-1], 
        target=qHome
    )
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', block], ry.OT.eq, [1e1])
    komo.addObjective([], ry.FS.scalarProductXZ, ['l_gripper', block], ry.OT.eq, [1e1], [0])
    komo.addObjective([], ry.FS.scalarProductXY, ['l_gripper', block], ry.OT.eq, [1e1], [1])

    komo.addObjective([], ry.FS.distance, ['l_palm', block], ry.OT.eq, [1e1], [-.25])
    komo.addObjective([1], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1e1], [0, 0 , 1])
    # no table collision
    komo.addObjective([], ry.FS.distance, ['l_palm', 'table'], ry.OT.ineq, [1e1], [-0.01])

    ret = ry.NLP_Solver(komo.nlp(), verbose=4) .solve()
    if prints == True:
        print(ret)
    
    q = komo.getPath()

    bot.moveTo(q[-1], 4)
    while bot.getTimeToEnd() > 0:
        bot.sync(C, .5)

def move_gripper_to(C, bot, add_frame_obj_str = 'target', carry_item = 'box1', orientation = 'vertical', prints=False):
    komo = ry.KOMO(C, 1, 1, 1, True)
    komo.addControlObjective([], 0, 0.1e1)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', add_frame_obj_str], ry.OT.eq, [1e1])
    
    if orientation == 'vertical':
        komo.addObjective([1], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1e1], [0, 0 , 1])    
        komo.addObjective([1], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, [1e1], [0, 1, 0])     

    if orientation == 'horizontal':
        komo.addObjective([1], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1e1], [1, 0, 0])
        komo.addObjective([1], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, [1e1], [0, 1, 0])     

    # no table collision
    komo.addObjective([], ry.FS.negDistance, ['l_palm', 'table'], ry.OT.ineq, [1e1], [-0.300])
    
    # avoid palm collision with box
    komo.addObjective([], ry.FS.distance, ['l_gripper', carry_item], ry.OT.ineq, [1e1], [-.001])
    

    # avoid box collision during transport of box
    boxes = [f"box{i}" for i in range(1, 4)]
    boxes.remove(carry_item)
    for box in boxes:
        komo.addObjective([], ry.FS.distance, [carry_item, box], ry.OT.ineq, [1e1], [-.001])


    ret = ry.NLP_Solver(komo.nlp(), verbose=4) .solve()
    if prints == True:
        print(ret)
    
    q = komo.getPath()
    
    bot.moveTo(q[-1], 4)
    while bot.getTimeToEnd() > 0:
        bot.sync(C, .1)

def retract_gripper(C, bot, distance = 0.25, orientation = 'vertical',prints = False):
    bot.sync(C,.5)
    gripper_pos = C.getFrame("l_gripper").getPosition()
    retract_to_pos = gripper_pos + [0.0, 0.0, distance]
    C.addFrame('retract_to_pos').setShape(ry.ST.marker, [.1, .1, .1]).setPosition(retract_to_pos).setColor([1, .5, 1])

    komo = ry.KOMO(C, 1, 1, 1, True)
    komo.addControlObjective([], 0, 0.1e1)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'retract_to_pos'], ry.OT.eq, [1e3])

    # orient block vertical
    if orientation == 'vertical':
        komo.addObjective([1], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1e1], [0, 0 , 1])    
        komo.addObjective([1], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, [1e1], [0, 1, 0])
        pass
    elif orientation == 'horizontal':
        komo.addObjective([1], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1e1], [1, 0, 0])
        komo.addObjective([1], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, [1e1], [0, 1, 0]) 
             

    ret = ry.NLP_Solver(komo.nlp(), verbose=4) .solve()
    if prints == True:
        print(ret)
    
    q = komo.getPath()
    
    bot.moveTo(q[-1], 4)
    while bot.getTimeToEnd() > 0:
        bot.sync(C, .1)



