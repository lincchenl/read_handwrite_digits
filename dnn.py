# module dnn.py
import numpy as np
import pickle

def normflat(data):
	remap=data.flatten()
	mean=np.mean(remap)
	deviation=np.std(remap)
	remap=(remap-mean)/deviation
	return remap

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
		self.result=self.gamma*self.x_norm+self.beta

	def backward(self,dout): #dout is the total m output diffs
		x_mu=self.x-self.x_mean
		std_inv=1./np.sqrt(self.x_std+self.epsilon)
		self.dx_norm = np.multiply(dout,self.gamma)
		self.dx_std = np.sum(self.dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
		self.dx_mean = np.sum(self.dx_norm * -std_inv, axis=0) + self.dx_std * np.mean(-2. * x_mu, axis=0)
		self.dx = (self.dx_norm * std_inv) + (self.dx_std * 2 * x_mu / self.batch_cnt) + (self.dx_mean / self.batch_cnt)
		self.dgamma = np.sum(dout * self.x_norm, axis=0)
		self.dbeta = np.sum(dout, axis=0)

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
	fwd_flag = False
	#backward
	dx_norm = None
	dx_std = None
	dx_mean = None
	dbeta = None
	dgamma = None
	dx = None
	bwd_flag = False

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
		self.result=self.gamma*self.x_norm+self.beta
		self.fwd_flag = True

	def backward(self,dout): #dout is the total m output diffs
		x_mu=self.x-self.x_mean
		std_inv=1./np.sqrt(self.x_std+self.epsilon)
		self.dx_norm = np.multiply(dout,self.gamma)
		self.dx_std = np.sum(self.dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
		self.dx_mean = np.sum(self.dx_norm * -std_inv, axis=0) + self.dx_std * np.mean(-2. * x_mu, axis=0)
		self.dx = (self.dx_norm * std_inv) + (self.dx_std * 2 * x_mu / self.batch_cnt) + (self.dx_mean / self.batch_cnt)
		self.dgamma = np.sum(dout * self.x_norm, axis=0)
		self.dbeta = np.sum(dout, axis=0)
		self.bwd_flag = True

class DO_layer:
	length = 0
	dp_rate = 0
	dp_array = None
	def __del__(self):
		self.dp_array = None
	def __init__(self, rate,cnt):
		self.dp_rate=rate
		self.dp_array=np.ones(cnt,dtype=np.float)
	def forward(self,x):
		self.dp_array=np.random.binomial(1, self.dp_rate, size=self.length)
		return np.multiply(self.dp_array,x)
	def backward(self,dout):
		return np.multiply(self.dp_array,dout)

class nn_layer:
	layer = int(0)
	count = int(0)
	inited = False
	bNorm = None
	dOut = None

	def __init__(self, count, p=None, c=None):
		self.data = np.zeros(count, dtype=np.float)
		self.partial = np.zeros(count, dtype=np.float)
		self.count = count
		self.parent = p
		self.child = c
		self.inited = True

	def __del__(self):
		self.child = None
		self.parent = None
		self.bNorm = None
		self.dOut = None

	def append(self, p=None):
		if isinstance(p, nn_layer):
			self.child = p.child
			self.parent = p
			p.child = self
			if p.layer == -1:
				p.layer = p.parent.layer + 1
			self.layer = p.layer + 1
			if not self.child:
				self.layer = -1
			# 分配空间存储参数
			self.para = np.zeros([self.count, p.count + 1], dtype=np.float)
			self.diff = np.zeros([self.count], dtype=np.float)
			self.dpara = np.zeros(self.para.shape, dtype=np.float)
			# adam
			self.mt = 0.
			self.vt = 0.
			# dropout
			self.dprate = 1.  # 默认禁用dropout

	def __selu(self, x):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.))

	def __selup(self, x):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * np.where(x > 0.0, 1., alpha * np.exp(x))

	def __relu(self, x):
		return np.where(x < 0., 0., x)

	def __relup(self, x):
		return np.where(x < 0., 0., 1.)

	def __softmax(self, x):
		e_x=np.exp(x)
		return e_x/e_x.sum()

	def __softmaxp(self, x):
		return np.ones(x.shape, dtype=np.float)

	def backward_1step(self, result):
		my_cnt = self.data.size
		parent_cnt = self.parent.data.size
		if not self.child:
			self.diff = result - self.data
		else:
			child_cnt = self.child.data.size
			self.diff.fill(0.)
			for j in range(child_cnt):
				self.diff[:] += self.child.diff[j] * self.child.para[j, :my_cnt]
		var_x = self.parent.data
		self.dpara[:, parent_cnt] = self.diff[:] * self.partial[:]
		for j in range(parent_cnt):
			self.dpara[:, j] = self.dpara[:, parent_cnt] * var_x[j]

	def forward_1step(self, train):
		parent_cnt = self.parent.data.size
		lresult = np.dot(self.para[:, :parent_cnt], self.parent.data) + self.para[:, parent_cnt]
		if not self.child:
			result = self.__softmax(lresult)
			self.partial = self.__softmaxp(lresult)
		else:
			result = self.__selu(lresult)
			self.partial = self.__selup(lresult)
		# dropout
		#if train:
		#	dropout = np.random.binomial(1, self.dprate, size=self.data.shape)
		#	np.multiply(result,dropout,out=result)
		#	np.multiply(self.partial,dropout,out=self.partial)
		#else:
		#	np.multiply(result, self.dprate, out=result)
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

	def train_input(self, map_to_train, result_no):
		remap=normflat(map_to_train)
		self.top.set_data(remap)
		result = np.zeros(self.tail.data.shape, dtype=np.float)
		result[result_no] = 1.0
		self.forward(True)
		self.backward(result)
		return sum(self.tail.diff ** 2) / 2.

	def para_commit(self, cat, ratio):
		# apply the ADAM algorithm to speedup the converge of the SGD
		p = self.tail
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		while p.parent:
			gt = p.dpara
			alpha = ratio / float(cat)
			p.mt = beta1 * p.mt + (1. - beta1) * gt
			p.vt = beta2 * p.vt + (1. - beta2) * gt ** 2
			mt_mean = p.mt / (1. - beta1)
			vt_mean = p.vt / (1. - beta2)
			gt.fill(0.)
			p.set_para(p.para + mt_mean / np.sqrt(vt_mean + epsilon) * alpha)
			p = p.parent

	def test(self, map_tt):
		remap=normflat(map_tt)
		self.top.set_data(remap)
		self.forward(False)

	def clear(self):
		p = self.top.child
		while p:
			p.mt.fill(0.)
			p.vt.fill(0.)
			p = p.child

	def forward(self, train):
		p = self.top.child
		while p:
			p.forward_1step(train)
			p = p.child

	def backward(self, result):
		p = self.tail
		while p.parent:
			p.backward_1step(result)
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
