import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as plotimg

def out_x_y(matrix):
    x = matrix[0]/matrix[2]
    y = matrix[1]/matrix[2]
    # print(x,y)
    return x, y

def angle_with_x_axis(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def distance_from_C_to_line_AB(A,B,C):
    numerator=abs((B[1]-A[1])*C[0]-(B[0]-A[0])*C[1]+B[0]*A[1]-B[1]*A[0])
    denominator=math.sqrt((B[1]-A[1]) ** 2+(B[0]-A[0]) ** 2)
    return numerator/denominator

def find_perpendicular_point_D(A,B,C):
    A=np.array(A)
    B=np.array(B)
    C=np.array(C)
    # 計算向量 AB 和 AC
    AB=B-A
    AC=C-A
    # 計算 t，使得 D = A + t * AB 是 C 到 AB 的垂足
    t=np.dot(AC,AB)/np.dot(AB,AB)
    # 計算點 D 的座標
    D=A+t*AB
    return D

def rotate_point(x,y,theta_deg):
    theta_rad=math.radians(theta_deg)
    x_new=x*math.cos(theta_rad)-y*math.sin(theta_rad)
    y_new=x*math.sin(theta_rad)+y*math.cos(theta_rad)
    return x_new, y_new

matrix = np.array([ [1.4237457108004437, -0.717919420091444, -403.8109154095238],
                    [0.0009548131181493375, 0.19893812144029333, 16],
                    [0.00005967581988433359, -0.000782903852698234, 1]])

xyp = [[ 922, 381],
      [1377, 226],
      [1900, 344],
      [1506, 604]]

xy =[]
car=[960,1080]
img = plotimg.imread('44.jpg')

try:
    inverse_matrix = np.linalg.inv(matrix)
    for i in xyp:
        rd = out_x_y(np.matmul(inverse_matrix, [i[0], i[1], 1]).tolist())
        xy.append(rd)

    # print(xy)
    
    
except np.linalg.LinAlgError:
    print("error!!")


D=find_perpendicular_point_D((xy[0][0],1080-xy[0][1]),(xy[3][0],1080-xy[3][1]),(car[0],car[1]))
# print(f'點D座標:({D[0]:>7.2f},{1080-D[1]:>7.2f})')

delta_y=math.sqrt((D[0]-car[0]) ** 2+(D[1]-car[1]) ** 2)/6
# print('delta_y=',delta_y)

delta_phi = -angle_with_x_axis((xy[0][0], 1080 - xy[0][1]), (xy[3][0], 1080 - xy[3][1]))
# print('delta_phi=', delta_phi)

x_new,y_new=rotate_point(D[0]-(xy[0][0]+xy[3][0])/2,D[1]-(1080-xy[0][1]+1080-xy[3][1])/2,delta_phi)
delta_x=x_new/6
# print('delta_x=',delta_x)

car[0],car[1]=out_x_y(np.matmul(inverse_matrix, [car[0],car[1], 1]).tolist())
# print(f'車子座標：({car[0]:>7.2f},{car[1]:>7.2f})')


h, w = img.shape[:2]
size = (w, h)  

# 做透視轉換
warped = cv2.warpPerspective(img, np.linalg.inv(matrix), size)

warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
plt.imshow(warped_rgb)
plt.title("Image after perspective transformation")
plt.axis('off')
plt.show()

# 要標的點
points = np.vstack((xy, [ [car[0], car[1]] ]))
# 顯示圖片
plt.imshow(warped_rgb)
for i, (x, y) in enumerate(points):
    plt.plot(x, y, 'ro')  # 標紅點
    plt.text(x + 10, y - 10, f'P{i}', color='green', fontsize=12)
plt.title("Image after perspective transformation")
plt.axis('off')
plt.show()

# 結果
print(f'車子座標 :({car[0]:>7.2f},{car[1]:>7.2f})')

for i in range(len(xy)):
    print(f'第{i:1d}點座標：({xy[i][0]:>7.2f},{xy[i][1]:>7.2f})')

print('delta_x=',delta_x)
print('delta_y=',delta_y)
print('delta_phi=',delta_phi)