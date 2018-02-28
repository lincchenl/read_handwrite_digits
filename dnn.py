# module dnn.py
import numpy as np
import pickle

class BN_layer:
	length = int(0)
	batch_cnt = int(0)
	epsilon = 1e-8
	gamma = None
	beta = None
	#foward
	x = None
	x_mean = None
	x_std = None
	x_norm = None
	result = None
	#backward
	dx_norm = None
	dx_std = None
	dx_mean = None
	dbeta = None
	dgamma = None
	dx = None

	def __del__(self):
		self.gamma = None
		self.beta = None
		# foward
		self.x = None
		self.x_mean = None
		self.x_std = None
		self.x_norm = None
		self.result = None
		# backward
		self.dx_norm = None
		self.dx_std = None
		self.dx_mean = None
		self.dbeta = None
		self.dgamma = None
		self.dx = None

	def __init__(self, length,cnt):
		self.length=length
		self.batch_cnt=cnt
		self.gamma=np.ones(length,dtype=np.float)
		self.beta=np.zeros(length,dtype=np.float)

	def forward(self,x):
		self.x=x
		self.x_mean=np.mean(self.x,axis=0)
		self.x_std=np.var(self.x,axis=0)
		self.x_norm=(self.x-self.x_mean)/np.sqrt(self.x_std+self.epsilon)
		self.result=self.x_norm*np.tile(self.gamma,x.shape[1]).reshape(x.shape)
		self.result+=np.tile(self.beta,x.shape[1]).reshape(x.shape)

	def backward(self,dout): #dout is the total m output diffs
		x_mu=self.x-self.x_mean
		std_inv=1./np.sqrt(self.x_std+self.epsilon)
		self.dx_norm = np.multiply(dout,np.tile(self.gamma,dout.shape[1]).reshape(dout.shape))
		self.dx_std = np.sum(self.dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
		self.dx_mean = np.sum(self.dx_norm * -std_inv, axis=0) + self.dx_std * np.mean(-2. * x_mu, axis=0)
		self.dx = (self.dx_norm * std_inv) + (self.dx_std * 2 * x_mu / self.batch_cnt) + (self.dx_mean / self.batch_cnt)
		self.dgamma = np.sum(dout * self.x_norm, axis=1)
		self.dbeta = np.sum(dout, axis=1)

class DO_layer:
	length = 0
	dp_rate = 0
	dp_array = None
	def __del__(self):
		self.dp_array = None
	def __init__(self, rate,cnt):
		self.dp_rate=rate
		self.length=cnt
	def forward(self,x):
		self.dp_array=np.random.binomial(1, self.dp_rate, size=self.length)
		return np.multiply(self.dp_array,x)
	def backward(self,dout):
		return np.multiply(self.dp_array,dout)

class AC_func:
	fNo = 0
	x = None
	def __init__(self,kind):
		self.fNo=kind

	def __selu(self, x):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.))

	def __selup(self):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * np.where(self.x > 0.0, 1., alpha * np.exp(self.x))
	def __relu(self, x):
		return np.where(x < 0., 0., x)

	def __relup(self):
		return np.where(self.x < 0., 0., 1.)
	def __softmax(self, x):
		e_x=np.exp(x)
		return e_x/e_x.sum()
	def __softmaxp(self):
		return np.ones(self.x.shape, dtype=np.float)

	def forward(self,x):
		switch = {
			1: self.__selu,
			2: self.__relu,
			3: self.__softmax
		}
		self.x=x
		return switch[self.fNo](x)

	def backward(self,dout):
		switch = {
			1: self.__selup,
			2: self.__relup,
			3: self.__softmaxp
		}
		return switch[self.fNo]()*dout


class nn_layer:
	layer = int(0)
	count = int(0)
	inited = False
	bNorm = None
	dOut = None
	aFunc = None

	def __init__(self, count, p=None, c=None):
		self.data = np.zeros(count, dtype=np.float)
		self.count = count
		self.parent = p
		self.child = c
		self.inited = True
		self.diff = np.empty([self.count], dtype=np.float)

	def __del__(self):
		self.child = None
		self.parent = None
		self.bNorm = None
		self.dOut = None
		self.aFunc = None

	def append(self, p=None):
		if isinstance(p, nn_layer):
			self.child = p.child
			self.parent = p
			p.child = self
			self.layer = p.layer + 1
			# 分配空间存储参数
			self.para = np.empty([self.count, p.count + 1], dtype=np.float)
			self.dpara = np.empty([self.count, p.count + 1], dtype=np.float)
			# adam
			self.mt = 0.
			self.vt = 0.

	# 每一层的处理顺序是 x->BN->全连接->激活函数->dropout->y
	# forward的时候就是按照这样传递data
	# backward的时候刚好顺序是相反的，传递的是diff
	def backward_1step(self, result,BN_No):
		my_cnt = self.data.size
		cd=self.child
		if not cd:
			# 输出层
			self.diff = result - self.data
		else:
			# 通过child的diff来求本层的diff
			child_cnt=cd.diff.size
			dout=cd.diff
			# dropout
			if cd.dOut :
				dout=cd.dOut.backward(dout)
			# 激活函数
			if cd.aFunc:
				dout=cd.aFunc.backward(dout)
			self.diff.fill(0.)
			for j in range(child_cnt):
				self.diff[:] += dout[j] * cd.para[j, :my_cnt]
			# Batch Normalization在循环外完成
			# 计算dpara
			if cd.bNorm:
				var_x = cd.bNorm.result[:,BN_No]
			else:
				var_x = self.data
			cd.dpara[:, my_cnt] = dout
			for j in range(my_cnt):
				cd.dpara[:, j] = cd.dpara[:, my_cnt] * var_x[j]


	def forward_1step(self, train,BN_No):
		parent_cnt = self.parent.data.size
		# Batch Normalization在循环外完成
		if self.bNorm:
			input=self.bNorm.result[:,BN_No]
		else :
			input=self.parent.data
		# 全连接层
		result = np.dot(self.para[:, :parent_cnt], input) + self.para[:, parent_cnt]
		# 激活函数
		if self.aFunc:
			result = self.aFunc.forward(result)
		# dropout层
		if self.dOut:
			if train:
				result=self.dOut.forward(result)
			else:
				result=self.dOut.dp_rate
		self.set_data(result)

	def set_para(self, data):
		if hasattr(self, 'para'):
			if data.size == self.para.size:
				self.para = data.copy()

	def set_data(self, data):
		if hasattr(self, 'data'):
			if data.size == self.data.size:
				self.data = data.copy()


class nn:
	layers = int(0)
	inited = False

	def __init__(self, top_cnt):
		self.layers = 1
		self.top = nn_layer(top_cnt)
		self.tail = self.top
		self.inited = True

	def addlayer(self, count):
		self.layers = self.layers + 1
		nn_layer(count).append(self.tail)
		self.tail = self.tail.child

	def save_params(self, filename):
		f = open(filename, 'wb')
		pickle.dump(self, f)
		f.close()

	def load_params(self, filename):
		f = open(filename, 'rb')
		obj = pickle.load(f)
		f.close()
		return obj

	def locate(self, layer):
		if layer in range(self.layers):
			if layer == 0:
				return self.top
			result = self.top
			for i in range(1, layer + 1):
				result = result.child
			return result
		else:
			return None

	def train_prepare(self):
		p = self.locate(1)
		while p:
			pshape = p.para.shape
			p.set_para(np.random.uniform(0.01 * 0.01, 0.02 * 0.02, pshape))
			p = p.child

	def para_commit(self, cat, ratio):
		# apply the ADAM algorithm to speedup the converge of the SGD
		p = self.tail
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		while p.parent:
			#adam
			gt = p.dpara
			alpha = ratio / float(cat)
			p.mt = beta1 * p.mt + (1. - beta1) * gt
			p.vt = beta2 * p.vt + (1. - beta2) * gt ** 2
			mt_mean = p.mt / (1. - beta1)
			vt_mean = p.vt / (1. - beta2)
			gt.fill(0.)
			p.set_para(p.para + mt_mean / np.sqrt(vt_mean + epsilon) * alpha)
			p = p.parent

	def copy_para_dnn(self, src):
		p = self.top.child
		q = src.top.child
		while p:
			p.para = q.para  # 各个计算dpara的子线程都对para只读不写，所以干脆让所有para都指向原始的引用
			p = p.child
			q = q.child

	# 消除nn_layer的循环引用使其能够被垃圾回收
	def __del__(self):
		self.layers = 0
		self.top = None
		self.tail = None
