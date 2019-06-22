import os
import argparse
import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry, \
                        get_anno_tracking, get_anno_odometry


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Monocular VO',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('seq', help='Input Seq for tracking',
                        default='0007', type=str)
    parser.add_argument('--path', help='Path of input info for tracking',
                        default='/home/EricHu/Workspace/kitti_tracking/training/', type=str)
    parser.add_argument('--save_name', help='Filename to save the trajectory',
                        type=str)
    parser.add_argument('--use_abs_scale', dest='use_abs_scale',
                        help='Estimate ego-motion using VO',
                        default=False, action='store_true')
    parser.add_argument('--skip_frame', dest='skip_frame', type=int, 
                        default=1,
                        help='Skip every N frame to test VO')
    args = parser.parse_args()
    return args



class Whiteboard(object):

    def __init__(self):
        self.board = np.zeros((600,600,3), dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.text = "{} Coordinates: x={:0.2f}m y={:0.2f}m z={:0.2f}m"
        self.position = {'pred': (20, 40), 'true': (20, 60)}
        self.size = {'pred': 1, 'true': 2}

    def draw(self, img_id, loc_type, location, color):
        x, y, z = location
        text = self.text.format(loc_type, x, y, z)
        cv2.rectangle(self.board, (0, 0), (80, 24), (0, 0, 0), -1)
        cv2.putText(self.board, '{:04d}'.format(img_id), (10, 20), 
                self.font, 1.5, (255,255,255), 2, 8)
        self._draw(x, y, text, color, loc_type)

    def _draw(self, x, y, text, color, loc_type):
        position = self.position[loc_type.lower()]
        size = self.size[loc_type.lower()]
        draw_loc = (int(x) + self.board.shape[0]//2,
                    int(y) + self.board.shape[0]//2) 
        cv2.rectangle(self.board, 
                    (0, position[1]-12), 
                    (self.board.shape[1], position[1]+2), 
                    (0, 0, 0), 
                    -1)
        cv2.putText(self.board, text, position, 
                self.font, 1.2, (255,255,255), 1, 8)
        cv2.circle(self.board, draw_loc, 1, color, size)

    def show(self, img=None):
        if img is not None: cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', self.board)
        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            exit()

    def save(self, save_name):
        cv2.imwrite(save_name, self.board)

def main():
    args = parse_args()

    image_path = os.path.join(args.path, 'image_02/', args.seq)
    oxts_path = os.path.join(args.path, 'oxts/', args.seq + '.txt')

    itms = [os.path.join(image_path, it) 
                for it in os.listdir(image_path) 
                    if it.endswith('.png')]
    annotation = get_anno_tracking(oxts_path)

    img_shape = cv2.imread(itms[0], 0).shape
    cam = PinholeCamera(img_shape[1], img_shape[0], 
                        718.8560, 718.8560, 
                        607.1928, 185.2157)
    vo = VisualOdometry(cam, annotation, args.use_abs_scale, args.skip_frame)

    wb = Whiteboard()

    for img_id in range(0, len(itms), args.skip_frame):
        img = cv2.imread(args.path + 'image_02/' + args.seq + '/' + str(img_id).zfill(6) + '.png', 0)

        vo.update(img, img_id)

        if(img_id > 0):
            cur_t = vo.cur_t.squeeze()
            x, y, z = cur_t[0], cur_t[2], -cur_t[1]
        else:
            x, y, z = 0., 0., 0.

        t_loc = np.array([vo.trueX, vo.trueY, vo.trueZ])
        t_color = (0, 0, 255)
        p_loc = np.array([x, y, z])
        p_color = (img_id*255.0/len(itms),255-img_id*255.0/len(itms),0)

        wb.draw(img_id, 'True', t_loc, t_color)
        wb.draw(img_id, 'Pred', p_loc, p_color)

        wb.show(img)

    if args.save_name:
        wb.save(args.save_name)


if __name__ == '__main__':
    main()
