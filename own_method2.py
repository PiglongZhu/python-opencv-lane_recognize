import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video = cv.VideoCapture('lane2.mp4')

def ROI(img):
    area = np.array([[(90,530),(390,350),(570,350),(890,530)]])
    mask = np.zeros_like(img)
    cv.fillPoly(mask,area,255)
    output = cv.bitwise_and(img,mask)
    return output

while (1):
    img = video.read()[1]
    # img = cv.imread('lane.jpg')
    cv.imwrite('test1.jpg',img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #获得感兴趣区域
    ROIimg = ROI(gray)
    #双边滤波
    blur = cv.bilateralFilter(ROIimg,5,150,150)
    #canny边缘
    canny = cv.Canny(blur,100,150)
    #霍夫线检测
    lines = cv.HoughLinesP(canny,1,np.pi/180,10,minLineLength=10,maxLineGap=10)
    #筛选直线（通过斜率和截距）
    L_lines_box = []
    R_lines_box = []
    for line in lines:
        [[x1,y1,x2,y2]] = line
        k = (y2-y1)/(x2-x1)
        b = y1 - x1*k
        if -1.5<k<-0.4 and b > 600:
            L_lines_box.append(line)
        if 0.4<k<1.5 and 900*k+b > 550:
            R_lines_box.append(line)

    #平均直线的斜率和截距
    L_k_all = []
    L_b_all = []
    for line in L_lines_box:
        [[x1, y1, x2, y2]] = line
        L_k_all.append((y2-y1)/(x2-x1))
        L_b_all.append(y1 - x1*(y2-y1)/(x2-x1))
    R_k_all = []
    R_b_all = []
    for line in R_lines_box:
        [[x1, y1, x2, y2]] = line
        R_k_all.append((y2-y1)/(x2-x1))
        R_b_all.append(y1 - x1*(y2-y1)/(x2-x1))


    #画出直线
    #求出平均斜率和截距
    L_k = np.mean(L_k_all)
    L_b = np.mean(L_b_all)
    R_k = np.mean(R_k_all)
    R_b = np.mean(R_k_all)
    #y = 350和535时
    xl1 = int((350 - L_b)/L_k)
    xl2 = int((535 - L_b)/L_k)
    xr1 = int((350 - R_b)/R_k)
    xr2 = int((535 - R_b)/R_k)
    #画出线
    # cv.line(img,(xl1,350),(xl2,535),(0,0,255),2,lineType=cv.LINE_AA)
    # cv.line(img,(xr1,350),(xr2,535),(0,0,255),2,lineType=cv.LINE_AA)
    #填充车道
    area = np.array([[[xl1,350],[xl2,535],[xr2,535],[xr1,350]]])
    img2 = img.copy()
    cv.fillPoly(img2,area,(0,255,0))
    img_f = cv.addWeighted(img,0.5,img2,0.5,0)

    # plt.imshow(img)
    # plt.show()

    cv.imshow('result',img_f)
    cv.waitKey(1)


cv.destroyAllWindows()
