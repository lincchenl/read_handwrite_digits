#module dnn.py
import numpy as np
from abc import ABCMeta, abstractmethod

class mnn:
	input = None # dimension [N,D]
	output = None # dimension [N,D]
	layers = int(0)
	head = None
	tail = None
	diff = None
	def __init__(self,icnt,ocnt,batch_cnt):
		self.input=np.zeros([batch_cnt,icnt],dtype=np.float)
		self.output = np.zeros([batch_cnt, ocnt], dtype=np.float)
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
	def forward(self,train,result=None):
		p=self.top
		while p :
			if p.parent:
				p.input=p.parent.output
			else:
				p.input = self.input
			p.forward(train)
			p=p.child
		self.output=self.tail.output
		if train:
			self.diff=result-self.output
	def backward(self):
		p=self.tail
		while p :
			if p.child:
				p.dout=p.child.dx
			else:
				p.dout=self.diff
			p.backward()
			p.update() #更新每一层参数，比如全连接层的系数或者BN层的beta，gamma
			p=p.parent

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
	def forward(self,train):
		pass
	@abstractmethod
	def backward(self):
		pass
	@abstractmethod
	def update(self):
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
		return np.ones(self.input.shape, dtype=np.float)

	def forward(self,train=True):
		switch = {
			1: self.__selu,
			2: self.__relu,
			3: self.__softmax
		}
		self.output=switch[self.__fNo](self.input)

	def backward(self):
		switch = {
			1: self.__selup,
			2: self.__relup,
			3: self.__softmaxp
		}
		self.dx=switch[self.__fNo](self.input)*self.dout

	def update(self):
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
	def forward(self,train):
		if train:
			self.dp_array=np.empty(self.input.shape,dtype=np.float)
			for i in range(self.input.shape[0]):
				self.dp_array[i,:]=np.random.binomial(1, self.dp_rate, size=self.input.shape[0])
			self.output=self.dp_array*self.input
		else:
			self.output=self.input*self.dp_rate
	def backward(self):
		self.dx=self.dout*self.dp_array
	def update(self):
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

	def forward(self,train):
		if train:
			if not (self.beta and self.gamma):
				self.beta=np.zeros(self.input.shape[1],dtype=np.float)
				self.gamma = np.ones(self.input.shape[1], dtype=np.float)
			self.x_mean=np.mean(self.input,axis=0)
			self.x_std=np.var(self.input,axis=0)
			self.x_norm=(self.input-self.x_mean)/np.sqrt(self.x_std+self.epsilon)
			self.output=self.x_norm*self.gamma
			self.output+=self.beta
		else:
			self.output=self.input

	def backward(self):
		x_mu=self.input-self.x_mean
		std_inv=1./np.sqrt(self.x_std+self.epsilon)
		self.dx_norm = self.dout*self.gamma
		self.dx_std = np.sum(self.dx_norm * x_mu, axis=0)
		self.dx_std *= -.5
		self.dx_std *= std_inv ** 3
		self.dx_mean = -np.sum(self.dx_norm *std_inv, axis=0)
		self.dx_mean += self.dx_std * np.mean(-2. * x_mu, axis=0)
		self.dx = (self.dx_norm * std_inv)
		self.dx += (self.dx_std * 2 * x_mu / self.batch_cnt)
		self.dx += (self.dx_mean / self.batch_cnt)
		self.dgamma = np.sum(self.dout * self.x_norm, axis=0)
		self.dbeta = np.sum(self.dout, axis=0)

	def update(self):
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
	def forward(self,train):
		cnt=self.input.shape[0]
		if not self.para:
			self.para=np.random.uniform(0.01 * 0.01, 0.02 * 0.02,[self.my_cnt,self.parent_cnt+1])
		self.output=np.empty([cnt,self.my_cnt],dtype=np.float)
		for i in range(cnt):
			self.output[i,:]=  np.dot(self.para[:, :self.parent_cnt], input)
			self.output[i, :] += self.para[:, self.parent_cnt]
	def backward(self):
		cnt = self.dout.shape[0]
		self.dpara=np.empty([cnt,self.my_cnt,self.parent_cnt+1],dtype=np.float)
		self.dx=np.zeros([cnt,self.parent_cnt],dtype=np.float)
		# 计算dx
		for i in range(cnt):
			for j in range(self.my_cnt):
				self.dx[i,:] += self.dout[i,j] * self.para[j, :self.parent_cnt]
		# 通过dx和input来计算dpara
		self.dpara[:,:,self.parent_cnt]=self.dout
		for i in range(self.parent_cnt):
			self.dpara[:,:,i] = self.dpara[:,:,self.parent_cnt] * self.input[:,i]
	def update(self):
		self.opti.input = self.para
		self.opti.dx = np.average(self.dpara,axis=0)
		self.opti.update()
		self.para=self.opti.output

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
		if not self.mt:
			self.mt=np.zeros(self.input.shape,dtype=np.float)
		if not self.vt:
			self.vt=np.zeros(self.input.shape,dtype=np.float)
		gt = self.input
		alpha = ratio
		self.mt = beta1 * self.mt + (1. - beta1) * gt
		self.vt = beta2 * self.vt + (1. - beta2) * gt ** 2
		mt_mean = self.mt / (1. - beta1)
		vt_mean = self.vt / (1. - beta2)
		self.output=self.input+mt_mean / np.sqrt(vt_mean + epsilon) * alpha



