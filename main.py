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
		nn1.test(datas[index, :, :])
		if nn1.tail.data[results[index]] > 0.5:
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
	img=np.empty((28*28,fp_cnt),dtype=np.float)
	for i in range(fp_cnt):
		img[:,i]=normflat(fp_img[i,:,:])


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
				result = np.zeros([10,cat],dtype=np.float)
				for j in sgd:
					# 新建cat个nn网络
					nns.append(creatnn())
					nns[k].copy_para_dnn(nn1)
					# 准备好cat个输入
					nns[k].top.set_data(img[:,j])
					result[fl_idx[j],k] = 1.0
					k = k + 1
				#开始训练
				#forward
				for j in range(1,nn1.layers):
					p=nn1.locate(j)
					if p.bNorm:
						#在推进之前计算本层的BN
						x=np.empty([p.parent.data.size,cat],dtype=np.float)
						for k in range(cat):
							x[:,k]=nns[k].locate(j).parent.data.copy()
						p.bNorm.forward(x)
					for k in range(cat):
						nns[k].locate(j).forward_1step(True,k)
				#backward
				for j in range(nn1.layers-1,-1,-1):
					p = nn1.locate(j)
					for k in range(cat):
						nns[k].locate(j).backward_1step(result[:, k], k)
					if p.bNorm:
						# 准备dout
						my_cnt = p.data.size
						parent_cnt = p.parent.data.size
						dout = np.empty([parent_cnt, cat], dtype=np.float)
						for k in range(cat):
							child_dout = nns[k].locate(j).diff
							# dropout
							if nns[k].locate(j).dOut:
								child_dout = nns[k].locate(j).dOut.backward(child_dout)
							# active func
							if nns[k].locate(j).aFunc:
								child_dout = nns[k].locate(j).aFunc.backward(child_dout)
							# 全连接
							dout[:,k].fill(0.)
							for l in range(my_cnt):
								dout[:, k] += child_dout[l] * nns[k].locate(j).para[l, :parent_cnt]
						p.bNorm.backward(dout)
				# 误差统计
				for k in range(cat):
					rs[k]=np.sum(nns[k].tail.diff**2)/2.
				# 累加结果
				for j in range(1,nn1.layers):
					nn1.locate(j).dpara.fill(0.)
					for k in range(1,nn1.layers):
						nn1.locate(k).dpara += nns[j].locate(k).dpara / cat
				# gamma和beta的学习
				#bn1.beta -= bn1.dbeta * 0.01
				#bn1.gamma -= bn1.dgamma * 0.01
				#bn2.beta -= bn2.dbeta * 0.01
				#bn2.gamma -= bn2.dgamma * 0.01
				#bn3.beta -= bn3.dbeta * 0.01
				#bn3.gamma -= bn3.dgamma * 0.01
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
		img1[:,i]=normflat(fp1_img[i,:,:])

	print("总测试样本的个数是：", fl1_cnt, "。")
	print("迭代100次后的准确率是：", test_accu("d:\\paras\\parad100.pkl", fl1_cnt, fp1_img, fl1_idx) * 100, "%  ")
	print("迭代1000次后的准确率是：", test_accu("d:\\paras\\parad1000.pkl", fl1_cnt, fp1_img, fl1_idx) * 100, "%  ")
	print("迭代2000次后的准确率是：", test_accu("d:\\paras\\parad2000.pkl", fl1_cnt, fp1_img, fl1_idx) * 100, "%  ")
	print("迭代5000次后的准确率是：", test_accu("d:\\paras\\parad5000.pkl", fl1_cnt, fp1_img, fl1_idx) * 100, "%  ")
	print("迭代10000次后的准确率是：", test_accu("d:\\paras\\parad10000.pkl", fl1_cnt, fp1_img, fl1_idx) * 100, "%  ")
	error = np.load("d:\\paras\\error.npy")
	plt.figure()
	plt.subplot(211)
	plt.plot(np.arange(float(group)), error)
	plt.subplot(212)
	plt.plot(np.arange(float(group)), np.log(error))
	plt.show()
