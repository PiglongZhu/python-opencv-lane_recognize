import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = 600
    y2 = int(image.shape[0]*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])



def averange_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    #获取每条直线的斜率和截距
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)#获得直线斜率和截距
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 and x1 < 600:
            left_fit.append((slope,intercept))
        elif slope > 0.4 and slope < 1:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit,axis=0)   #axis=0为列，1为行
    right_fit_average = np.average(right_fit, axis=0)  # axis=0为列，1为行
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    # print(left_line)
    return np.array([left_line,right_line])


def draw_area(names):
    point = []
    for name in names:
        x1,y1,x2,y2 = name.reshape(4)
        point.append([x1,y1])
        point.append([x2,y2])
    point_np = np.array([point[0], point[1], point[3], point[2]], np.int32)
    print(point_np)
    return point_np





def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 100, 150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(255,255,0),3)
    return line_image


def ROI(image):
    # height = image.shape[0]
    # width = image.shape[1]
    area1 = np.array([
        [(280,550),(420,550),(550,450),(480,450)]
                        ])
    area2 = np.array([[(660,580),(870,550),(650,440),(580,440)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask,[area1,area2],255)
    masked_image = cv.bitwise_and(image,mask)
    return masked_image


# image = cv.imread('test10.jpg')
# lane_image = image.copy()
# canny_image = canny(lane_image)
# cropped_image = ROI(canny_image)
# lines = cv.HoughLinesP(cropped_image,2,np.pi/180,3,np.array([]),minLineLength=3,maxLineGap=1)#霍夫找线
# line_image = display_lines(lane_image,lines)
# averaged_lines = averange_slope_intercept(lane_image,lines)#此时左右各剩一条平均后的直线
# line_image = display_lines(lane_image,averaged_lines)#画出直线
# combo = cv.addWeighted(lane_image,0.8,line_image,1,0)
# cv.imshow('result',combo)
# cv.imshow('2',line_image)
#
# print(lines[0])
# plt.imshow(canny_image)
# plt.show()


cap = cv.VideoCapture('road2.mp4')
while (cap.isOpened()):
    _,image = cap.read()
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = ROI(canny_image)
    # cv.imwrite('test10.jpg',lane_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=5)  # 霍夫找线
    averaged_lines = averange_slope_intercept(lane_image, lines)  # 此时左右各剩一条平均后的直线
    # line_image = display_lines(lane_image, averaged_lines)  # 画出直线

    point_all = draw_area(averaged_lines)
    cv.fillConvexPoly(image, point_all, (255, 255, 0))

    combo = cv.addWeighted(lane_image, 1, image, 0.5, 0)
    cv.imshow('result', combo)
    if cv.waitKey(1) == 27:
        break
cap.release()



cv.waitKey(0)
cv.destroyAllWindows()