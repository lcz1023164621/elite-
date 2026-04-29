#!/usr/bin/env python3
import rclpy
from elite_msgs.srv import CartMove, JointMove, StopMove
from elite_msgs.srv import AddPathPoint, ClearPathPoint, MoveByPath

class EliteArmMove:
    def __init__(self, ):
        self.get_logger().info("ready to move arm")
        self.cart_mov_srv = self.create_service(CartMove, 'cart_move_server', self.cart_move_cb)
        self.joint_mov_srv = self.create_service(JointMove, 'joint_move_server', self.joint_move_cb)
        self.stop_mov_srv = self.create_service(StopMove, 'stop_move_server', self.stop_move_cb)
        self.add_path_point_srv = self.create_service(AddPathPoint, 'add_path_point_server', self.add_path_point_cb)
        self.clear_path_point_srv = self.create_service(ClearPathPoint, 'clear_path_point_server', self.clear_path_point_cb)
        self.move_by_path_srv = self.create_service(MoveByPath, 'move_by_path_server', self.move_by_path_cb)

    def cart_move_cb(self, request, response):
        target_joint =  request.target_joint.tolist()
        speed = request.speed
        speed_type = request.speed_type
        acc = request.acc
        dec = request.dec        
        is_block = request.is_blocking
        result_ = self.elite_robot.move_line(  # pylint: disable=E1101
            target_joint, speed, speed_type, acc, dec)
        if is_block:
            self.elite_robot.wait_stop()  # pylint: disable=E1101
        response.result = result_
        return response

    def joint_move_cb(self, request, response):
        joint_point_ = request.target_joint.tolist()
        speed_ = request.speed
        acc_ = request.acc
        dec_ = request.dec
        is_block_ = request.is_blocking
        result_ = self.elite_robot.move_joint(  # pylint: disable=E1101
            joint_point_, speed_, acc_, dec_)
        if is_block_:
            self.elite_robot.wait_stop()  # pylint: disable=E1101
        if type(result_) == bool:
            response.result = result_
        else:
            response.result = result_[0]
        return response

    def stop_move_cb(self, request, response):
        self.get_logger().info("stop_move_server recieved command to stop the robot")
        result_ = self.elite_robot.stop()  # pylint: disable=E1101
        response.result = result_
        # ToDo: Log the response
        return response 

    def add_path_point_cb(self, request, response):
        waypoint = request.way_point.tolist()
        move_type = request.move_type
        speed_type = request.speed_type
        speed = request.speed
        acc = request.acc
        dec = request.dec
        smooth = request.smooth
        circular_radius = request.circular_radius
        cond_type = request.cond_type
        cond_num = request.cond_num
        cond_value = request.cond_value
        result_ = self.elite_robot.add_path_point(waypoint, move_type, speed, acc, dec, smooth, circular_radius, 
        speed_type, cond_type, cond_num, cond_value)
        response.result = result_
        return response

    def clear_path_point_cb(self, request, response):
        result_ = self.elite_robot.clear_path_point()
        response.result = result_
        return response

    def move_by_path_cb(self, request, response):
        result_ = self.elite_robot.move_by_path()
        if (result_ == -1):
            response.result = False
        else:
            response.result = True
        return response