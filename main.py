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


# initiate a nn
def creatnn():
	nn_object = dnn.nn(28 * 28)
	nn_object.addlayer(100)
	nn_object.addlayer(100)
	nn_object.addlayer(10)
	nn_object.locate(1).dprate=0.8
	nn_object.locate(2).dprate=0.8
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
	fp = np.fromfile(path_images, dtype=np.uint8)
	fp_mac, fp_cnt, fp_row, fp_col = struct.unpack(">IIII", fp[0:16])
	fp_img = np.asarray(fp[16:])
	fp_img.resize([fp_cnt, fp_row, fp_col])
	# category
	no_idx = np.zeros((fl_cnt, 10), dtype=np.int)
	index = np.zeros(10, dtype=np.int)
	for i in range(fl_cnt):
		j = int(fl_idx[i])
		no_idx[index[j], j] = i
		index[j] += 1
	group = np.min(index)

	nn1 = creatnn()
	nn1.train_prepare()
	cat = 100
	group = 10000
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
				# cleanup
				# the cleanup work has been done int he para_commit seciton
				for j in sgd:
					# 此处循环可以用多个线程来实现

					tmp_nn = creatnn()
					tmp_nn.copy_para_dnn(nn1)

					rs[k] = tmp_nn.train_input(fp_img[j, :, :], fl_idx[j])
					k = k + 1
					# 累加结果
					for j in range(1, nn1.layers):
						nn1.locate(j).dpara += tmp_nn.locate(j).dpara / cat
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
