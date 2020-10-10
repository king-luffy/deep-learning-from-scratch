#导入pickle模块
import pickle

#打开一个名为data1.pkl的文件，打开方式为二进制读取(参数‘rb’)
file_to_read = open('sample_weight.pkl', 'rb')

#通过pickle的load函数读取data1.pkl中的对象，并赋值给data2
data2 = pickle.load(file_to_read)

#打印data2
print(data2)

#关闭文件对象
file_to_read.close()