import pickle

f = open('model/cnn.pkl','rb')
#使用load的方法将数据从pkl文件中读取出来
result = pickle.load(f)
print(result)
#关闭文件
f.close()





