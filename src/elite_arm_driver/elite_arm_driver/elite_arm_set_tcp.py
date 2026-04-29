#!/usr/bin/env python3
import rclpy
from elite_msgs.srv import SetTCP

class EliteArmSetTCP():
    def __init__(self):
        self.get_logger().info("starting elite arm set_TCP Service")
        self.set_tcp_srv = self.create_service(SetTCP, 'set_tcp', self.set_tcp_cb)

    def set_tcp_cb(self, request, response):
        if (request.index > 7):
            self.get_logger().info("set_TCP Service: request.index is out of range [0, 7]")
            response.result = False
            return response

        if (request.unit_type != 0 and request.unit_type != 1):
            self.get_logger().info("set_TCP Service: request.unit_type is out of range [0|1]")
            response.result = False
            return response

        if (request.offset_x < -500.0 or request.offset_x > 500.0 or
            request.offset_y < -500.0 or request.offset_y > 500.0 or
            request.offset_z < -500.0 or request.offset_z > 500.0):
            self.get_logger().info("set_TCP Service: request.offset is out of range [-500.0, 500.0]")
            response.result = False
            return response

        if ((request.unit_type == 0 and (request.rx < -180.0 or request.rx > 180.0 or
                                        request.ry < -180.0 or request.ry > 180.0 or
                                        request.rz < -180.0 or request.rz > 180.0)) or
            (request.unit_type == 1 and (request.rx < -3.14159 or request.rx > 3.14159 or
                                        request.ry < -3.14159 or request.ry > 3.14159 or
                                        request.rz < -3.14159 or request.rz > 3.14159))):
            self.get_logger().info("set_TCP Service: request.rotation is out of range [-180, 180.0] (deg) or [-π, π] (rad)")
            response.result = False
            return response

        result = self.elite_robot.send_CMD("cmd_set_tcp", {"point": [request.offset_x, request.offset_y, request.offset_z, request.rx, request.ry, request.rz], "tool_num": request.index, "unit_type": request.unit_type})  # pylint: disable=E1101

        response.result = result
        print(f"result:{result}")
        return response

