import numpy as np
import json

import cv2
import png

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

class rgbd():

    def __init__(self, pxToMetre, focalx, focaly, cx, cy):
        self.pxToMetre = pxToMetre
        self.focalx = focalx
        self.focaly = focaly
        self.cx = cx
        self.cy = cy
        
    def pt3d_from_depth(self, u, v, d):

        d *= self.pxToMetre
        x = (self.cx-u) / self.focalx
        y = (self.cy-v) / self.focaly
        z = d / np.sqrt(1. + x**2 + y**2)
        x = x*z
        y = y*z

        return x,y,z

    def pt3d_cloud_from_depth(self, depth_image):

        pt3d_cloud = []

        for u in range(depth_image.shape[1]):
            for v in range(depth_image.shape[0]):

                d = depth_image[v,u]
                x,y,z = self.pt3d_from_depth(u,v,d)
                pt3d_cloud.append(x)
                pt3d_cloud.append(y)
                pt3d_cloud.append(z)
        
        pt3d_cloud = np.asarray(pt3d_cloud).reshape((-1,3))

        return pt3d_cloud

    def pt3d_cloud_color_from_RGB(self, RGB_image):

        pt3d_cloud_color = []

        for u in range(RGB_image.shape[1]):
            for v in range(RGB_image.shape[0]):

                x = RGB_image[v,u,2]
                y = RGB_image[v,u,1]
                z = RGB_image[v,u,0]
                pt3d_cloud_color.append(x)
                pt3d_cloud_color.append(y)
                pt3d_cloud_color.append(z)
        
        pt3d_cloud_color = np.asarray(pt3d_cloud_color).reshape((-1,3))

        return pt3d_cloud_color

if __name__ == '__main__':

    index = 20

    path_RGB_image = './frame_{}.jpg'.format(str(index).zfill(5))
    RGB_image = cv2.imread(path_RGB_image)

    cv2.imshow('RGB_image', RGB_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('RGB_image...')
    print('RGB_image.shape')
    print(RGB_image.shape)
    print()

    path_depth_image = './depth_{}.png'.format(str(index).zfill(5))
    depth_image = cv2.imread(path_depth_image, cv2.IMREAD_UNCHANGED)

    print('depth_image...')
    print('dtype of depth_image')
    print(depth_image.dtype)
    print('max and min of data')
    print(np.max(depth_image.flatten()), np.min(depth_image.flatten()))
    print('depth_image.shape')
    print(depth_image.shape)
    print()

    ratio_depth_to_RGB = depth_image.shape[0] / RGB_image.shape[0]
    RGB_image = cv2.resize(RGB_image, (depth_image.shape[1], depth_image.shape[0]), interpolation = cv2.INTER_AREA)

    path_frame_json = './frame_{}.json'.format(str(index).zfill(5))
    with open(path_frame_json) as f:
        frame_json = json.load(f)

    print('frame_json', frame_json)
    for k,v in frame_json.items():
        print(k)
    print('intrinsics...')
    print(np.asarray(frame_json['intrinsics']).reshape((3,3)))
    print()

    intrinsics = np.asarray(frame_json['intrinsics']).reshape((3,3))
    focalx = intrinsics[0,0] * ratio_depth_to_RGB
    focaly = intrinsics[1,1] * ratio_depth_to_RGB
    cx = intrinsics[0,2] * ratio_depth_to_RGB
    cy = intrinsics[1,2] * ratio_depth_to_RGB
    pxToMetre = 0.001

    rgbd_0 = rgbd(pxToMetre,focalx,focaly,cx,cy)

    pt3d_cloud = rgbd_0.pt3d_cloud_from_depth(depth_image)
    pt3d_cloud_color = rgbd_0.pt3d_cloud_color_from_RGB(RGB_image)

    create_output(pt3d_cloud, pt3d_cloud_color, './{}.ply'.format(str(index).zfill(5)))