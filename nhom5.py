import csv
import numpy as np
import math
from numpy import *
import pandas as pd

path = "D:/BTL_AI/UngDungDuBaoThoiTiet/UngDungDuBaoThoiTiet/"

def loadData(): 
    f = open(path + "ThoiTiet_dulieu.csv")
    data = csv.reader(f) #csv format
    data = np.array(list(data))# convert to matrix
    data = np.delete(data, 0, 0)# delete header
    np.random.shuffle(data) # shuffle data
    f.close()
    trainSet = data[:362] #training data from 1->362
    testSet = data[362:]# the others is testing data
    return trainSet, testSet

def calcDistancs(pointA, pointB, numOfFeature=5): 
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)

def kNearestNeighbor(trainSet, point, k): 
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1], #nhãn
            "value": calcDistancs(item, point) #khoảng cách 2 điểm
        })
    distances.sort(key=lambda x: x["value"]) # sắp xếp dựa trên value
    labels = [item["label"] for item in distances] #lấy các nhãn của bộ dữ liệu
    return labels[:k]  

def findMostOccur(arr): 
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num  #nếu số lượng nhãn này lớn hơn số lượng nhãn lớn nhất trước đó tìm được thì gán lại
            ans = label
    return ans # đưa ra nhãn xuất hiện nhiều nhất

def loadDataInput(path):
    f = open(path, "r")
    data = csv.reader(f) 
    data = np.array(list(data))
    data = np.delete(data, 0, 0) #xóa header
    f.close()
    return data[:1]

if __name__ == "__main__":
    
    with open(path + "ThoiTiet_daxuly.csv",mode="w") as f:
        trainSet, testSet = loadData() #đọc đưa ra bộ train và test
        numOfRightAnswer = 0 #số lượng nhãn dự đoán đúng
        writer = csv.writer(f)
        #way to write to csv file
        writer.writerow(['Label', 'Predicted'])
        for item in testSet: #chạy từng dữ liệu trong bộ test
            knn = kNearestNeighbor(trainSet, item, 5) #tính k hàng xóm của bộ train gần dữ liệu này nhất 
            answer = findMostOccur(knn) #đưa ra nhãn xuất hiện nhiều nhất
            numOfRightAnswer += item[-1] == answer
            writer.writerow([item[-1], answer])
    print("\nAccuracy = ", numOfRightAnswer/len(testSet))

    f.close()
    
    var_tempMax = input('\nNhap nhiet do cao nhat: ')  #đoạn nhập values mà muốn dự đoán
    var_tempMin = input('Nhap vao nhiet do thap nhat: ')
    var_wind = input('Nhap vao toc do gio: ')
    var_cloud = input('Nhap vao luong may: ')
    var_rel = input('Nhap vao do am: ')

    data = {
        "Max Temperature" : [var_tempMax],
        "Min Temperature" : [var_tempMin],
        "Wind Speed" : [var_wind],
        "Loud Cover" : [var_cloud],
        "Relative Humidity" : [var_rel]
    }
    df = pd.DataFrame(data)
    df.to_csv(path + "ThoiTiet_input.csv", index=False) #tao file csv chua dl vua nhap
    
    f = open(path + "ThoiTiet_testInput.csv", "w")
    writer = csv.writer(f) #way to write to csv file
    item = loadDataInput(path + "ThoiTiet_input.csv") #đọc file input đó
    for i in item:
        knn = kNearestNeighbor(trainSet, i, 5)
        answer = findMostOccur(knn)
        writer.writerow([i[-1], answer])
        print("predicted: {}".format(answer)) #đưa ra dự đoán