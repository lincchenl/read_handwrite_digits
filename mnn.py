#module mnn.py
import numpy as np
from abc import ABCMeta, abstractmethod
import pickle

class mnn:
	input = None # dimension [N,D]
	output = None # dimension [N,D]
	layers = int(0)
	head = None
	tail = None
	diff = None

	def addlayer(self,object):
		self.layers=self.layers+1
		if not self.head:
			self.head=object
			object.parent=None
		else:
			self.tail.child=object
			object.parent=self.tail
		self.tail=object
		object.child=None
	def forward(self,train=True,result=None):
		p=self.head
		k=0
		while p :
			if p.parent:
				p.input=p.parent.output
			else:
				p.input = self.input
			p.forward(train)
			#if isinstance(p,full_connect):
			#	print ("第",k,"层全连接：",np.average(np.abs(p.para)),np.average(np.abs(p.output)))
			p=p.child
			k+=1
		self.output=self.tail.output
		if train:
			self.diff=result-self.output
	def backward(self,train=True):
		p=self.tail
		while p :
			if p.child:
				p.dout=p.child.dx
			else:
				p.dout=self.diff
			p.backward(train)
			p.update(train) #更新每一层参数，比如全连接层的系数或者BN层的beta，gamma
			p=p.parent
	def save(self, filename):
		#在保存之前先把网络的规格变成batch=1,瘦身
		self.input=self.input[0:0,:]
		self.forward(False)
		self.backward(False)
		f = open(filename, 'wb')
		pickle.dump(self, f)
		f.close()
	def load(self, filename):
		f = open(filename, 'rb')
		obj = pickle.load(f)
		f.close()
		return obj

# 每一层网络的抽象类，它包括输入输出，以及dx和dout四个成员数据 以及forward和backward和update三个成员函数待实现
class layer:
	__metaclass__ = ABCMeta
	input = None
	output = None
	dx = None
	dout = None
	child = None
	parent = None

	def __del__(self):
		self.input=None
		self.output=None
		self.dx=None
		self.dout=None
		self.child=None
		self.parent=None

	@abstractmethod
	def forward(self,train=True):
		pass
	@abstractmethod
	def backward(self,train=True):
		pass
	@abstractmethod
	def update(self,train=True):
		pass
# 激活函数类
class active_function(layer):
	__fNo = 0
	def __init__(self,kind):
		self.__fNo=kind

	def __selu(self, x):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.))
	def __selup(self,x):
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
	def __softmaxp(self,x):
		return 1.

	def forward(self,train=True):
		switch = {
			1: self.__selu,
			2: self.__relu,
			3: self.__softmax
		}
		cnt=self.input.shape[0]
		self.output=np.zeros(self.input.shape,dtype=np.float)
		for i in range(cnt):
			self.output[i,:]=switch[self.__fNo](self.input[i,:])

	def backward(self,train=True):
		if train:
			switch = {
				1: self.__selup,
				2: self.__relup,
				3: self.__softmaxp
			}
			cnt = self.dout.shape[0]
			self.dx = np.zeros(self.dout.shape, dtype = np.float)
			for i in range(cnt):
				self.dx[i,:] = switch[self.__fNo](self.input[i,:]) * self.dout[i,:]
		else:
			self.dx=None
			self.dout=None

	def update(self,train=True):
		pass
# Dropout类
class dropout(layer):
	dp_rate = 0
	dp_array = None
	def __del__(self):
		layer.__del__(self)
		self.dp_array = None
	def __init__(self, rate):
		self.dp_rate=rate
	def forward(self,train=True):
		if train:
			self.dp_array=np.empty(self.input.shape,dtype=np.float)
			for i in range(self.input.shape[0]):
				self.dp_array[i,:]=np.random.binomial(1, self.dp_rate, size=self.input.shape[1])
			self.output=self.dp_array*self.input
		else:
			self.output=self.input*self.dp_rate
	def backward(self,train=True):
		if train:
			self.dx=self.dout*self.dp_array
		else:
			self.dx=None
			self.dout=None
	def update(self,train=True):
		pass
# Batch_normalization类
class batch_normalization(layer):
	epsilon = 1e-8
	gamma = None
	beta = None
	#foward
	x_mean = None
	x_std = None
	x_norm = None
	#backward
	dx_norm = None
	dx_std = None
	dx_mean = None
	dbeta = None
	dgamma = None
	#运行时均值和方差
	beta1 = 0.9
	beta2 = 0.99
	running_mean = 0
	running_std = 0

	def __del__(self):
		layer.__del__(self)
		self.gamma = None
		self.beta = None
		# foward
		self.x_mean = None
		self.x_std = None
		self.x_norm = None
		# backward
		self.dx_norm = None
		self.dx_std = None
		self.dx_mean = None
		self.dbeta = None
		self.dgamma = None

	def forward(self,train=True):
		if train:
			if self.beta is None:
				self.beta=np.zeros(self.input.shape[1],dtype=np.float)
			if self.gamma is None:
				self.gamma = np.ones(self.input.shape[1], dtype=np.float)
			self.x_mean=np.mean(self.input,axis=0)
			self.x_std=np.std(self.input,axis=0)
			self.x_norm=(self.input-self.x_mean)/(self.x_std+self.epsilon)
			self.output=self.x_norm*self.gamma
			self.output+=self.beta
			self.running_mean = self.beta1*self.running_mean+(1-self.beta1)*self.x_mean
			self.running_std = self.beta2*self.running_std+(1-self.beta2)*self.x_std
		else:
			x_norm=(self.input-self.running_mean)/(self.running_std+self.epsilon)
			self.output=x_norm*self.gamma+self.beta

	def backward(self,train=True):
		if train:
			batch_cnt=self.dout.shape[0]
			x_mu=self.input-self.x_mean
			std_inv=1./np.sqrt(self.x_std+self.epsilon)
			self.dx_norm = self.dout*self.gamma
			self.dx_std = np.sum(self.dx_norm * x_mu, axis=0)
			self.dx_std *= -.5
			self.dx_std *= std_inv ** 3
			self.dx_mean = -np.sum(self.dx_norm *std_inv, axis=0)
			self.dx_mean += self.dx_std * np.mean(-2. * x_mu, axis=0)
			self.dx = (self.dx_norm * std_inv)
			self.dx += (self.dx_std * 2 * x_mu / batch_cnt)
			self.dx += (self.dx_mean / batch_cnt)
			self.dgamma = np.sum(self.dout * self.x_norm, axis=0)
			self.dbeta = np.sum(self.dout, axis=0)
		else:
			self.dx=None
			self.dout=None
			self.dpara=None
			self.dbeta=None
			self.dgamma=None

	def update(self,train=True):
		if train:
			self.beta += self.dbeta
			self.gamma += self.dgamma
# 全连接类
class full_connect(layer):
	para=None
	dpara=None
	opti=None
	parent_cnt=0
	my_cnt=0
	def __init__(self,parent_cnt,my_cnt):
		self.parent_cnt=parent_cnt
		self.my_cnt=my_cnt
	def __del__(self):
		layer.__del__(self)
		self.para=None
		self.opti=None
		self.dpara=None
	def forward(self,train=True):
		cnt=self.input.shape[0]
		if self.para is None:
			self.para=np.random.uniform(0.01 * 0.01, 0.02 * 0.02,[self.my_cnt,self.parent_cnt+1])
		self.output=np.empty([cnt,self.my_cnt],dtype=np.float)
		for i in range(cnt):
			self.output[i,:]=  np.dot(self.para[:, :self.parent_cnt], self.input[i,:])
			self.output[i, :] += self.para[:, self.parent_cnt]
	def backward(self,train=True):
		if train:
			cnt = self.dout.shape[0]
			self.dpara=np.empty([cnt,self.my_cnt,self.parent_cnt+1],dtype=np.float)
			self.dx=np.zeros([cnt,self.parent_cnt],dtype=np.float)
			# 计算dx
			for i in range(cnt):
				for j in range(self.my_cnt):
					self.dx[i,:] += self.dout[i,j] * self.para[j, :self.parent_cnt]
			# 通过dx和input来计算dpara
			self.dpara[:,:,self.parent_cnt]=self.dout.copy()
			for i in range(self.parent_cnt):
				for j in range(cnt):
					self.dpara[j,:,i] = self.dout[j,:] * self.input[j,i]
		else:
			self.dx=None
			self.dout=None
	def update(self,train=True):
		if train:
			self.opti.input = self.para
			self.opti.dx = np.average(self.dpara,axis=0)
			self.opti.update()
			self.para=self.opti.output.copy()

# 迭代优化的抽象类
class optimization:
	__metaclass__ = ABCMeta
	input=None
	dx=None
	output=None
	ratio=0.01

	def __del__(self):
		self.input=None
		self.dx=None
		self.output=None

	@abstractmethod
	def update(self):
		pass
class batch(optimization):
	def __init__(self,ratio):
		self.ratio=ratio
	def update(self):
		self.output=self.input+self.dx*self.ratio
class adam(optimization):
	mt=None
	vt=None
	epsilon=1e-8
	beta1=0.9
	beta2=0.999
	def __init__(self,ratio):
		self.ratio=ratio
	def __del__(self):
		self.mt=None
		self.vt=None
	def update(self):
		if self.mt is None:
			self.mt=np.zeros(self.input.shape,dtype=np.float)
		if self.vt is None:
			self.vt=np.zeros(self.input.shape,dtype=np.float)
		gt = self.dx
		alpha = self.ratio
		self.mt = self.beta1 * self.mt + (1. - self.beta1) * gt
		self.vt = self.beta2 * self.vt + (1. - self.beta2) * gt ** 2
		mt_mean = self.mt / (1. - self.beta1)
		vt_mean = self.vt / (1. - self.beta2)
		self.output=self.input+mt_mean / np.sqrt(vt_mean + self.epsilon) * alpha



