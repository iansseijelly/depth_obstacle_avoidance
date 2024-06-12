# A test file testing BSM
import bsm 
import cv2

if __name__ == '__main__':
    oaa = bsm.ObstacleAvoidAgent()
    depth_image = cv2.imread("../output/img/c_disparity.png", cv2.IMREAD_GRAYSCALE)
    oaa.image_read(depth_image)
    # steer, state = oaa.steering(pitch_offset=0)
    # print("steering: {}".format('right' if steer > 0 else 'left'), steer)
    # print("state: {}".format(state))

    steer, state = oaa.steering_behavior(pitch_offset=0)
    print("behavior steering: {}".format('right' if steer > 0 else 'left'), steer)
    print("behavior state: {}".format(state))

    