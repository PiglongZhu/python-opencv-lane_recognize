import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
kernel = np.ones((3,3),np.uint8)

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = 550
    y2 = int(image.shape[0]*(3/5))
    print('y的范围',y1,'到',y2)
    x1 = int((y1-intercept)/slope)
    print('x1为',x1)
    x2 = int((y2-intercept)/slope)
    print('x2为',x2)
    return np.array([x1,y1,x2,y2])


def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 100, 150)
    return canny


def display_lines(image,lines):
    line_image = np.zeros_like(image)
    # print('line:',line)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv.line(line_image,(x1,y1),(x2,y2),(255,255,0),3)
    return line_image

def ROI(image):
    # height = image.shape[0]
    # width = image.shape[1]
    area1 = np.array([
        [(280,550),(420,550),(550,450),(490,450)]
                        ])
    area2 = np.array([[(660,580),(870,550),(650,440),(580,440)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask,[area1,area2],255)
    masked_image = cv.bitwise_and(image,mask)
    return masked_image


def nearest_line(image,lines):
    left_fit = [1]
    right_fit = [1]
    all_left_distance = [0]
    all_right_distance = [10000]
    #获取每条直线的斜率和截距
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1)#获得直线斜率和截距
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.5 and slope > -1.1:
                distance_left = (570-x1)+(570-x2)
                if distance_left > all_left_distance[0]:
                    all_left_distance.pop(0)
                    all_left_distance.append(distance_left)
                    left_fit.pop(0)
                    left_fit.append([slope,intercept])
            elif slope > 0 and slope >0.5:
                distance_right = (x1-570)+(x2-570)
                if distance_right < all_right_distance[0]:
                    all_right_distance.pop(0)
                    all_right_distance.append(distance_right)
                    right_fit.pop(0)
                    right_fit.append([slope,intercept])


    final_left = left_fit[0]#通过索引得到左侧最近线的斜率和截距[slope,intercept]
    print('左侧斜率和截距', final_left)
    left_line = make_coordinates(image,final_left)#画出线
    final_right = right_fit[0]#通过索引得到右侧最近线的斜率和截距
    print('右侧斜率和截距', final_right)
    right_line = make_coordinates(image,final_right)#画出线
    return np.array([[left_line],[right_line]])


# 测试调试的代码
# image = cv.imread('test2.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = ROI(canny_image)
# lines = cv.HoughLinesP(cropped_image, 1, np.pi / 180, 5, np.array([]), minLineLength=5, maxLineGap=1)  # 霍夫找线
# nearest_line1 = nearest_line(lane_image, lines)
# line_image = display_lines(lane_image, nearest_line1)
# combo = cv.addWeighted(lane_image, 0.8, line_image, 1, 0)
# print('两条线坐标：', nearest_line1)
# cv.imshow('result',combo)




cap = cv.VideoCapture('road2.mp4')
while (cap.isOpened()):
    _,image = cap.read()
    if _ == True:
        lane_image = np.copy(image)
        # cv.imwrite('test2.jpg',lane_image)  #测试时使用的，将有问题的那帧保存下来，进行调试
        canny_image = canny(lane_image)
        cropped_image = ROI(canny_image)
        lines = cv.HoughLinesP(cropped_image, 1, np.pi / 180, 5, np.array([]), minLineLength=5, maxLineGap=1)  # 霍夫变换找线
        nearest_line1 = nearest_line(lane_image, lines)
        line_image = display_lines(lane_image, nearest_line1)
        combo = cv.addWeighted(lane_image, 0.8, line_image, 1, 0)
        print('两条线坐标：', nearest_line1)
        cv.imshow('result',combo)
        if cv.waitKey(1) == 27:
            break
cap.release()








# image2 = cv.imread('test2.jpg')
# plt.imshow(image2)
# plt.show()


cv.waitKey(0)
cv.destroyAllWindows()
