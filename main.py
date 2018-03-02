import struct
import numpy as np
import dnn
import matplotlib.pyplot as plt

path_lables = ".\\source\\train-labels.idx1-ubyte"
path_images = ".\\source\\train-images.idx3-ubyte"

# 读取图片文件
def testndraw(data, nn_object):
	nn_object.test(data)
	plt.imshow(data, cmap='gray')
	plt.show()

# 对原始图片进行标准化处理
def normflat(data):
	remap=data.flatten()
	mean=np.mean(remap)
	deviation=np.std(remap)
	remap=(remap-mean)/deviation
	return remap

# initiate a nn
def creatnn():
	nn_object = dnn.nn(28 * 28)
	nn_object.addlayer(100)
	nn_object.locate(1).bNorm=bn1
	#nn_object.locate(1).dOut=dnn.DO_layer(0.8,100)
	nn_object.locate(1).aFunc=dnn.AC_func(2) #relu
	nn_object.addlayer(100)
	nn_object.locate(2).bNorm=bn2
	#nn_object.locate(2).dOut=dnn.DO_layer(0.8,100)
	nn_object.locate(2).aFunc=dnn.AC_func(2) #relu
	nn_object.addlayer(10)
	nn_object.locate(3).bNorm=bn3
	nn_object.locate(3).aFunc=dnn.AC_func(3) #softmax
	return nn_object


def test_accu(filename, count, datas, results):
	right = 0
	wrong = 0
	nn1=creatnn()
	nn1=nn1.load_params(filename)
	for index in range(count):
		nn1.test(datas[index,:])
		if nn1.tail.data[results[index]] > 0.5 :
			right += 1
		else:
			wrong += 1
	return right / (right + wrong)


# where the main program starts
if __name__ == "__main__":
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


	cat = 100
	group = 10000
	bn1 = dnn.BN_layer(28 * 28, cat)
	bn2 = dnn.BN_layer(100, cat)
	bn3 = dnn.BN_layer(100, cat)
	nn1 = creatnn()
	nn1.train_prepare()
	first = True
	error = np.zeros(group, dtype=float)
	rs = np.zeros(cat, dtype=float)
	if first:
		for i in range(group):
			res = 10.0
			cnt = 1
			# nn1.clear()
			if i == 100 - 1:
				nn1.save_params("d:\\paras\\parad100.pkl")
			if i == 1000 - 1:
				nn1.save_params("d:\\paras\\parad1000.pkl")
			if i == 2000 - 1:
				nn1.save_params("d:\\paras\\parad2000.pkl")
			if i == 5000 - 1:
				nn1.save_params("d:\\paras\\parad5000.pkl")
			if i == 10000 - 1:
				nn1.save_params("d:\\paras\\parad10000.pkl")

			sgd = np.random.randint(0, fl_cnt, size=cat)
			while res > 1e-5 and cnt < 2:
				rs.fill(0.)
				k = 0
				nns = []
				result = np.zeros([cat,10],dtype=np.float)
				for j in sgd:
					# 新建cat个nn网络
					nns.append(creatnn())
					nns[k].copy_para_dnn(nn1)
					# 准备好cat个输入
					nns[k].top.set_data(img[j,:])
					result[k,fl_idx[j]] = 1.0
					k = k + 1
				#开始训练
				#forward
				for j in range(1,nn1.layers):
					p=nn1.locate(j)
					if p.bNorm:
						#在推进之前计算本层的BN
						x=np.empty([cat,p.parent.data.size],dtype=np.float)
						for k in range(cat):
							x[k,:]=nns[k].locate(j).parent.data.copy()
						p.bNorm.forward(x)
					for k in range(cat):
						nns[k].locate(j).forward_1step(True,k)
				#计算误差
				for k in range(cat):
					diff=nns[k].tail.data-result[k,:]
					rs[k]=np.sum(diff**2)/2.
				#backward
				for j in range(nn1.layers-1,0,-1):
					p = nn1.locate(j)
					for k in range(cat):
						nns[k].locate(j).backward_1step(result[k,:], k)
					if p.bNorm:
						# 准备dout
						dout = np.empty([cat,nn1.locate(j).diff.size], dtype=np.float)
						for k in range(cat):
							dout[k,:]=nns[k].locate(j).diff
						p.bNorm.backward(dout)
				# 累加结果
				for j in range(1,nn1.layers):
					nn1.locate(j).dpara.fill(0.)
					for k in range(1,nn1.layers):
						nn1.locate(k).dpara += nns[j].locate(k).dpara / cat
				# gamma和beta的学习
				for j in range(1,nn1.layers):
					bn=nn1.locate(j).bNorm
					if bn :
						bn.beta += bn.dbeta
						bn.gamma += bn.dgamma
				# 更新全连接层的参数
				nn1.para_commit(cat, 0.1)
				res = sum(rs) / cat
				cnt += 1
				print(i, cnt - 1, res)
			error[i] = res
		np.save("d:\\paras\\error", error)
	# test trained network
	path_lables1 = ".\\source\\t10k-labels.idx1-ubyte"
	path_images1 = ".\\source\\t10k-images.idx3-ubyte"
	# 读取标签文件
	fl1 = np.fromfile(path_lables1, dtype=np.uint8)
	fl1_mgc, fl1_cnt = struct.unpack(">II", fl1[0:8])
	fl1_idx = np.asarray(fl1[8:])
	fp1 = np.fromfile(path_images1, dtype=np.uint8)
	fp1_mac, fp1_cnt, fp1_row, fp1_col = struct.unpack(">IIII", fp1[0:16])
	fp1_img = np.asarray(fp1[16:])
	fp1_img.resize([fp1_cnt, fp1_row, fp1_col])
	# 测试图片预处理
	img1=np.empty((28*28,fp_cnt),dtype=np.float)
	for i in range(fp1_cnt):
		img1[i,:]=normflat(fp1_img[i,:,:])

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
