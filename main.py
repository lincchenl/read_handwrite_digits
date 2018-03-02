import struct
import numpy as np
import mnn
import matplotlib.pyplot as plt

path_lables = ".\\source\\train-labels.idx1-ubyte"
path_images = ".\\source\\train-images.idx3-ubyte"
path_lables1 = ".\\source\\t10k-labels.idx1-ubyte"
path_images1 = ".\\source\\t10k-images.idx3-ubyte"

# 对原始图片进行标准化处理
def normflat(data):
	remap=data.flatten()
	#mean=np.mean(remap)
	#deviation=np.std(remap)
	#remap=(remap-mean)/np.sqrt(deviation)
	remap=remap/255
	return remap

# initiate a nn
def creatnn():
	nn=mnn.mnn()
	fc1=mnn.full_connect(28*28,100)
	fc2=mnn.full_connect(100,100)
	fc3=mnn.full_connect(100,10)
	fc1.opti=mnn.adam(0.1)
	fc2.opti=mnn.adam(0.1)
	fc3.opti=mnn.adam(0.1)
	nn.addlayer(fc1)
	nn.addlayer(mnn.batch_normalization())
	nn.addlayer(mnn.active_function(1))
	nn.addlayer(fc2)
	nn.addlayer(mnn.batch_normalization())
	nn.addlayer(mnn.active_function(1))
	nn.addlayer(fc3)
	nn.addlayer(mnn.active_function(3))
	return nn

def test_accu(filename, count, datas, results):
	right = 0
	wrong = 0
	nn1=mnn.mnn.load(filename)
	for index in range(count):
		nn1.input=datas[index:index,:]
		nn1.forward(False)
		if nn1.output[results[index]] > 0.5 :
			right += 1
		else:
			wrong += 1
	return right / (right + wrong)


# where the main program starts
if __name__ == "__main__":
	# 训练数据
	# 读取标签文件
	fl = np.fromfile(path_lables, dtype=np.uint8)
	fl_mgc, fl_cnt = struct.unpack(">II", fl[0:8])
	fl_idx = np.asarray(fl[8:])
	# 读取图片文件
	print ("读取文件中......")
	fp = np.fromfile(path_images, dtype=np.uint8)
	fp_mac, fp_cnt, fp_row, fp_col = struct.unpack(">IIII", fp[0:16])
	fp_img = np.asarray(fp[16:])
	fp_img.resize([fp_cnt, fp_row, fp_col])
	# 对图片进行预处理
	print("图片预处理......")
	img=np.empty((fp_cnt,28*28),dtype=np.float)
	for i in range(fp_cnt):
		img[i,:]=normflat(fp_img[i,:,:])

	# 验证数据
	fl1 = np.fromfile(path_lables1, dtype=np.uint8)
	fl1_mgc, fl1_cnt = struct.unpack(">II", fl1[0:8])
	fl1_idx = np.asarray(fl1[8:])
	fp1 = np.fromfile(path_images1, dtype=np.uint8)
	fp1_mac, fp1_cnt, fp1_row, fp1_col = struct.unpack(">IIII", fp1[0:16])
	fp1_img = np.asarray(fp1[16:])
	fp1_img.resize([fp1_cnt, fp1_row, fp1_col])
	# 测试图片预处理
	img1=np.empty((fp1_cnt,28*28),dtype=np.float)
	for i in range(fp1_cnt):
		img1[i,:]=normflat(fp1_img[i,:,:])

	cat = 100
	group = 10000
	nn1 = creatnn()
	first = True
	error = np.zeros(group, dtype=float)
	rs = np.zeros(cat, dtype=float)
	if first:
		for i in range(group):
			res = 10.0
			cnt = 1
			# nn1.clear()
			if i == 100 - 1:
				nn1.save("d:\\paras\\parad100.pkl")
			if i == 1000 - 1:
				nn1.save("d:\\paras\\parad1000.pkl")
			if i == 2000 - 1:
				nn1.save("d:\\paras\\parad2000.pkl")
			if i == 5000 - 1:
				nn1.save("d:\\paras\\parad5000.pkl")
			if i == 10000 - 1:
				nn1.save("d:\\paras\\parad10000.pkl")

			sgd = np.random.randint(0, fl_cnt, size=cat)
			while res > 1e-5 and cnt < 2:
				k = 0
				result = np.zeros([cat,10],dtype=np.float)
				for j in sgd:
					result[k,fl_idx[j]]=1
					k+=1
				#开始训练
				nn1.input=img[sgd,:]
				nn1.forward(True,result)
				#计算误差
				rs[:]=np.sum(nn1.diff**2./2.,axis=1)
				#backward
				nn1.backward()
				res = np.average(rs)
				cnt += 1
				print(i, cnt - 1, res)
			error[i] = res
		np.save("d:\\paras\\error", error)


	print("总测试样本的个数是：", fl1_cnt, "。")
	print("迭代100次后的准确率是：", test_accu("d:\\paras\\parad100.pkl", fl1_cnt, img1, fl1_idx) * 100, "%  ")
	print("迭代1000次后的准确率是：", test_accu("d:\\paras\\parad1000.pkl", fl1_cnt, img1, fl1_idx) * 100, "%  ")
	print("迭代2000次后的准确率是：", test_accu("d:\\paras\\parad2000.pkl", fl1_cnt, img1, fl1_idx) * 100, "%  ")
	print("迭代5000次后的准确率是：", test_accu("d:\\paras\\parad5000.pkl", fl1_cnt, img1, fl1_idx) * 100, "%  ")
	print("迭代10000次后的准确率是：", test_accu("d:\\paras\\parad10000.pkl", fl1_cnt, img1, fl1_idx) * 100, "%  ")
	error = np.load("d:\\paras\\error.npy")
	plt.figure()
	plt.subplot(211)
	plt.plot(np.arange(float(group)), error)
	plt.subplot(212)
	plt.plot(np.arange(float(group)), np.log(error))
	plt.show()
