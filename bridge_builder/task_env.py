import robotic as ry
import numpy as np

def create_n_boxes(C, N, position='fixed'):

    color_map = [[1,0,.5], # pink
                [0.5,1,0], # lime green
                [.5,1,1], # light turqois
                [0.5,0,1], # violet
                [1,0,0], # red
                [0,1,0], # green
                [0,0,1], # blue
                [1,.5,0], # orange
                [1,1,.5], # lime yellow
                [1,.5,1], # light pink                            
                ]

    for i in range(N):
        box_name = 'box{}'.format(i + 1)

        if position == 'fixed':
            position_val1 = 0.1 * (i - 5)
            C.addFrame(box_name) \
                .setPosition([position_val1, 0.05, 0.69]) \
                .setShape(ry.ST.ssBox, size=[0.04, 0.04, 0.12, 0.001]) \
                .setColor(color_map[i % len(color_map)]) \
                .setContact(True) \
                .setMass(1e-2)


        elif position == 'random':
            C.addFrame(box_name) \
                .setPosition([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.60]) \
                .setShape(ry.ST.ssBox, size=[0.05, 0.05, 0.12, 0.005]) \
                .setColor(color_map[i % len(color_map)]) \
                .setContact(True) \
                .setMass(1e-2)
    
    return C

def create_construction_site(C, position = 'fixed'):
    
    if position == 'fixed':
        target_location = C.addFrame('construction_site', 'table')
        target_location.setShape(ry.ST.box, size = [.275, .275, .090])
        target_location.setRelativePosition([-.5, -.225, .0])
        target_location.setColor([1., 0.75, 0.])
        return target_location
    
    elif position == 'random':
        target_location = C.addFrame('construction_site', 'table')
        target_location.setShape(ry.ST.box, size = [.3, .3, .1])
        target_location.setRelativePosition([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), .0])
        target_location.setColor([1, 0.75, 0])
        return target_location

def create_pillar_target_postion(C, position = 'fixed', n_pillars = 2):
    
    if position == 'fixed':
        for i in range(0, n_pillars):
            target_pillar = C.addFrame('pillar_target_pos' + str(i+1), 'table')
            target_pillar.setShape(ry.ST.box, size=[.065, .065, .090])
            target_pillar.setRelativePosition([-.45 - (i / 10) , -.225, .0])
            target_pillar.setColor([0,1,0])
    
    elif position == 'random':
        for i in range(0, n_pillars):
            C = C.addFrame('pillar_target_pos' + str(i+1), 'table')
            target_pillar.setShape(ry.ST.box, size=[.065, .065, .090])
            target_pillar.setRelativePosition([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), .0])
            target_pillar.setColor([0,1,0])
        return C
        