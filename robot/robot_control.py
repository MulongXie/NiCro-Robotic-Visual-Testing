import os
import sys

root = '/'.join(__file__.split('/')[:-1])
sys.path.append(root)
sys.path.append(os.path.join(root, 'uarm'))

from uarm.wrapper import SwiftAPI
from logzero import logger
import time


class RobotController(object):
    def __init__(self, port=None, speed=10000, press_distance=10):
        try:
            self.swift = SwiftAPI(port=port)
        except Exception as e:
            self.swift = None
        self.move_speed = speed
        self.press_distance = press_distance # 按压力度
        # self.swift.set_position(z=60, speed=100, wait=False, timeout=10, cmd='G1')

    def reset(self):
        logger.debug("Loading Robot Drivers...")
        self.swift.set_position(x=150, y=0, z=70, speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        # time.sleep(wait_time)

    def __click_before(self, point):
        self.swift.set_position(x=point[0], y=point[1], z=point[2], speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        self.swift.set_position(z=point[2] - 2, speed=self.move_speed, wait=False, timeout=10, cmd='G1')
        self.swift.flush_cmd()

    def click(self, coor):
        self.__click_before(coor)
        self.swift.set_position(x=coor[0], y=coor[1], z=coor[2] + 10, speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        logger.debug("Robot Click X: %d, Y:%d" % (coor[0], coor[1]))
        self.reset()

    def doubleclick(self, coor):
        self.__click_before(coor)
        self.swift.set_position(x=coor[0], y=coor[1], z=coor[2], speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        self.__click_before(coor)
        self.swift.set_position(x=coor[0], y=coor[1], z=coor[2], speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        self.reset()

    def longPress(self, coor):
        self.swift.set_position(x=coor[0], y=coor[1], z=coor[2], speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        time.sleep(1)
        self.swift.set_position(z=coor[2] + self.press_distance, speed=100, wait=False, timeout=10, cmd='G0')
        self.swift.flush_cmd()
        self.reset()

    def swipe(self, start_coord, end_coord, swipe_speed=50000):
        self.swift.set_position(x=start_coord[0], y=start_coord[1], z=start_coord[2], speed=self.move_speed, wait=False, timeout=10, cmd='G0')
        self.swift.set_position(x=end_coord[0], y=end_coord[1], z=end_coord[2], speed=swipe_speed, wait=False, timeout=10, cmd='G0')
        self.reset()

    def close_connect(self):
        self.swift.disconnect()

    def reconnect(self):
        self.swift.connect()
