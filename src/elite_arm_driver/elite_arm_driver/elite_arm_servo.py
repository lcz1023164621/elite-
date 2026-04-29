#!/usr/bin/env python3
import rclpy
from elite_msgs.srv import RobotServo, ClearAlarm

class EliteArmServo():
    def __init__(self):
        self.get_logger().info("starting elite arm servo service")
        self.servo_srv = self.create_service(RobotServo, 'servo', self.servo_cb)
        self.clear_alarm_arv = self.create_service(ClearAlarm, 'clear_alarm', self.clear_alarm_cb)

    def servo_cb(self, request, response):
        if (request.on_off == True):
            result = self.elite_robot.robot_servo_on()
        else:
            result = self.elite_robot.set_servo_status(0)  # pylint: disable=E1101
            if result == True:
                self.elite_robot.logger.debug("servo status set success")
        
        response.result = result
        print(f"result:{result}")
        return response

    def clear_alarm_cb(self, request, response):
        result = self.elite_robot.clear_alarm()
        response.result = result
        return response
        