import torch
import pickle
import time
import HELPER
import matplotlib.pyplot as plt



class DeepConvNet(object):

    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],#num_filters: List of length (L - 1) giving the number of convolutional filters to use in each macro layer.
                 max_pools=[0, 1, 2, 3, 4],#max_pools: List of integers giving the indices of the macro layers that should have max pooling (zero-indexed).
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 hidden_dims=[256, 256],
                 dropout_param={'p': 0.5, 'mode': 'train'},#p为drop的概率
                 dtype=torch.float,
                 device='cuda'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 3) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+3#修改全连接层数量先在这改
        #例如num_filters=[8, 8, 8, 8, 8]，表示有五个卷积层，每个卷积层有八个容器，这时num_layers=8
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.hidden_dims = hidden_dims
        self.dropout_param = dropout_param
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

       
        C, H, W = input_dims
        for i in range(len(num_filters)):
            if weight_scale == "kaiming":
                weight_scale = 2.0 / (C * 3 * 3)#kaiming初始化，以0为中心，2/n为方差，VGGNet中使用了3x3的卷积核
                self.params['W' + str(i + 1)] = kaiming_initializer(C, num_filters[i], 3, relu=True, device=device, dtype=dtype)
                #'W' + str(i + 1)表示'W1','W2'等,也可以通过self.params[f'W{i+1}']表示
            else:
                self.params['W' + str(i + 1)] = torch.randn(num_filters[i], C, 3, 3, dtype=dtype, device=device) * weight_scale    
            self.params['b' + str(i + 1)] = torch.zeros(num_filters[i], dtype=dtype, device=device)
            #每一个滤波器拥有一个bias，所以bias的数量等于滤波器的数量。相加时利用了广播机制
           
            if self.batchnorm:
                self.params['gamma' + str(i + 1)] = torch.ones(num_filters[i], dtype=dtype, device=device)
                self.params['beta' + str(i + 1)] = torch.zeros(num_filters[i], dtype=dtype, device=device)
            C = num_filters[i]#下一层的输入通道数等于上一层的输出通道数
            if i in max_pools:#List of integers giving the indices of the macro layers that should have max pooling (zero-indexed) 如果列表为[1, 3, 5]，则表示在网络的第2、第4和第6个层（以零为基础索引）应该进行最大池化操作
                H, W = H // 2, W // 2#池化层的stride为2，所以H和W都会减半
        if weight_scale == "kaiming":#从卷积到全连接的初始化
          #第一层全连接
          self.params['W' + str(len(num_filters) + 1)] = kaiming_initializer(C * H * W, self.hidden_dims[0], relu=True, device=device, dtype=dtype)
          #最后一层的输入通道数等于上一层的输出通道数乘以H和W，因为最后一层是全连接层，所以输入通道数等于H*W
          self.params['b' + str(len(num_filters) + 1)] = torch.zeros(self.hidden_dims[0], dtype=dtype, device=device)
          #第二层全连接
          self.params['W' + str(len(num_filters) + 2)] = kaiming_initializer(self.hidden_dims[0], self.hidden_dims[1], relu=True, device=device, dtype=dtype)
          self.params['b' + str(len(num_filters) + 2)] = torch.zeros(self.hidden_dims[1], dtype=dtype, device=device)
          #最后一层全连接
          self.params['W' + str(len(num_filters) + 3)] = kaiming_initializer(self.hidden_dims[1], num_classes, relu=False, device=device, dtype=dtype)
          self.params['b' + str(len(num_filters) + 3)] = torch.zeros(num_classes, dtype=dtype, device=device)
        else:
          #第一层全连接
          self.params['W' + str(len(num_filters) + 1)] = torch.randn(C * H * W, self.hidden_dims[0], dtype=dtype, device=device) * weight_scale
          self.params['b' + str(len(num_filters) + 1)] = torch.zeros(self.hidden_dims[0], dtype=dtype, device=device)
          #第二层全连接
          self.params['W' + str(len(num_filters) + 2)] = torch.randn(self.hidden_dims[0], self.hidden_dims[1], dtype=dtype, device=device) * weight_scale
          self.params['b' + str(len(num_filters) + 2)] = torch.zeros(self.hidden_dims[1], dtype=dtype, device=device)
          #最后一层全连接
          self.params['W' + str(len(num_filters) + 3)] = torch.randn(self.hidden_dims[1], num_classes, dtype=dtype, device=device) * weight_scale
          self.params['b' + str(len(num_filters) + 3)] = torch.zeros(num_classes, dtype=dtype, device=device)
        # When using batch normalization, there are scale and shift
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 6#三个全连接层，所以加6
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
      #forward pass
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        out = X
        caches = []
        for i in range(self.num_layers - 3):#self.num_layers = len(num_filters)+3
            if i in self.max_pools:
                if self.batchnorm:
                    out, cache = Conv_BatchNorm_ReLU_Pool.forward(out, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)], conv_param, self.bn_params[i],pool_param)
                else:
                    out, cache = Conv_ReLU_Pool.forward(out, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], conv_param, pool_param)
            else:
                if self.batchnorm:
                    out, cache = Conv_BatchNorm_ReLU.forward(out, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)], conv_param, self.bn_params[i])
                else: 
                    out, cache = Conv_ReLU.forward(out, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], conv_param)
            caches.append(cache)
        out1, cache = Linear_ReLU.forward(out, self.params['W' + str(self.num_layers-2)], self.params['b' + str(self.num_layers-2)])
        out1,cache_dropout = Dropout.forward(out1, self.dropout_param)
        caches.append((cache,cache_dropout))  #list.append()只能添加一个元素，如果添加的是一个元组，则整个元组作为一个元素添加到列表中     
        out2, cache = Linear_ReLU.forward(out1, self.params['W' + str(self.num_layers-1)], self.params['b' + str(self.num_layers-1)])
        out2,cache_dropout = Dropout.forward(out2, self.dropout_param)
        caches.append((cache,cache_dropout))
        scores, cache = Linear.forward(out2, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        caches.append(cache)
           
    
       
        #backward pass
        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers):
            W = self.params['W' + str(i + 1)]
            loss +=  self.reg * torch.sum(W * W)
        for i in range(self.num_layers,self.num_layers-3,-1):
           if i == self.num_layers:
               dout, grads['W' + str(i)], grads['b' + str(i)] = Linear.backward(dout, caches.pop())
           else:
               dout = Dropout.backward(dout, caches[i-1][1])#二维列表中append是竖向添加，1表示第二个元素 由于第一层对应cache第0行，所以i要减一
               dout, grads[f'W{i}'], grads[f'b{i}'] = Linear_ReLU.backward(dout, caches[i-1][0])
         
            
        for i in range(self.num_layers - 3, 0, -1):#范围为左闭右开，所以不包括0
            if i-1 in self.max_pools:#max_pools从0开始计数，故i需要减1
                if self.batchnorm:
                    dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = Conv_BatchNorm_ReLU_Pool.backward(dout, caches[i-1])
                else:
                    dout, grads['W' + str(i)], grads['b' + str(i)] = Conv_ReLU_Pool.backward(dout, caches[i-1])
            else:
                if self.batchnorm:
                    dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = Conv_BatchNorm_ReLU.backward(dout, caches[i-1])
                else:
                    dout, grads['W' + str(i)], grads['b' + str(i)] = Conv_ReLU.backward(dout, caches[i-1])
        for i in range(self.num_layers):
            grads['W' + str(i + 1)] += 2 * self.reg * self.params['W' + str(i + 1)]
          
        grads['x'] = dout

        return loss, grads


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            x_mean = torch.mean(x, dim=0)#dim=0表示对每一列求均值，x_mean的维度为1xD
            x_var = torch.var(x, dim=0,unbiased=False)#x_var的维度也为1xD,如果没有unbiased=False,则默认为无偏估计，即除以N-1，这里是除以N
            x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
            out = gamma * x_normalized + beta
            running_mean = momentum * running_mean + (1 - momentum) * x_mean
            running_var = momentum * running_var + (1 - momentum) * x_var
            cache = (x, x_mean, x_var, x_normalized, gamma, beta, eps)
    
        elif mode == 'test':
          
            x_normalized = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_normalized + beta
            
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        x, x_mean, x_var, x_normalized, gamma, beta, eps = cache
        N, D = x.shape
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(dout * x_normalized, dim=0)
        dx_normalized = dout * gamma
        dx_var = torch.sum(dx_normalized * (x - x_mean) * (-0.5) * (x_var + eps)**(-1.5), dim=0)
        dx_mean = torch.sum(dx_normalized * (-1) / torch.sqrt(x_var + eps), dim=0) + dx_var * torch.mean(-2 * (x - x_mean), dim=0)
        dx = dx_normalized / torch.sqrt(x_var + eps) + dx_var * 2 * (x - x_mean) / N + dx_mean/ N
        #中间的dx_var * 2 * (x - x_mean)的x_mean不用再对x求导是因为在求dx_mean的时候已经对x求导了，已经考虑过x对x_mean的影响就不用重复考虑


        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        x, x_mean, x_var, x_normalized, gamma, beta, eps = cache
        N, D = x.shape
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(dout * x_normalized, dim=0)
        dx_normalized = dout * gamma
        dx = (N * dx_normalized - torch.sum(dx_normalized, dim=0) - x_normalized * torch.sum(dx_normalized * x_normalized, dim=0)) / (N * torch.sqrt(x_var + eps))

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        #mode已经包含在bn_param中了，不用单独再考虑
        x_flat = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1).T#permute(1, 0, 2, 3)表示将原来的NCHW转换为CNHW，reshape(x.shape[1], -1)表示将原来的CNHW变为N*（C*H*W），T表示转置
        out_flat, cache = BatchNorm.forward(x_flat, gamma, beta, bn_param)
        out = out_flat.T.reshape(x.shape[1], x.shape[0], x.shape[2], x.shape[3]).permute(1, 0, 2, 3)
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        dout_flat = dout.permute(1, 0, 2, 3).reshape(dout.shape[1], -1).T
        dx_flat, dgamma, dbeta = BatchNorm.backward(dout_flat, cache)
        dx = dx_flat.T.reshape(dout.shape[1], dout.shape[0], dout.shape[2], dout.shape[3]).permute(1, 0, 2, 3)


        return dx, dgamma, dbeta

class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK. 
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
       
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * (gain / Din)**0.5#randn默认生成标准差为1，均值为0，所以需要乘以标准差的平方根
        
        
    else:
       
        weight = torch.randn(Dout, Din, K, K, dtype=dtype, device=device) * (gain / (Din * K * K))**0.5
        #Dout在前是因为卷积核的通道数需要等于输入的通道数，而第一维，也就是对应的输入维度的N的位置，表示输出的层数，初始化时传入num_filters[i]即可 
       
    return weight

class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
       
        x_reshaped=x.view(x.shape[0],-1)
        out = x_reshaped.mm(w) + b
       
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
       
        dx=dout.mm(w.t()).view(x.shape)
        dw=x.view(x.shape[0],-1).t().mm(dout)
        db=dout.sum(dim=0)
       
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
    
        out=x.clamp(min=0)
       
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        dx=(x>0).float()*dout
        
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    
    next_w = w - config['learning_rate'] * dw
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w += v
    
    config['velocity'] = v

    return next_w, config

def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = torch.zeros_like(w)
    
    config['t'] += 1
    config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dw
    mt = config['m'] / (1-config['beta1']**config['t'])
    config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dw*dw)
    vc = config['v'] / (1-(config['beta2']**config['t']))
    w = w - (config['learning_rate'] * mt)/ (torch.sqrt(vc) + config['epsilon'])
    next_w = w
    

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
           
            u1=(torch.rand(*x.shape)>p)/p
            u1=u1.to(x.device)
            mask=u1
            out=x*u1
           
        elif mode == 'test':
          
            
            out=x

          

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            
            dx=dout*mask
          
        elif mode == 'test':
            dx = dout
        return dx




def create_convolutional_solver_instance(data_dict, dtype, device):                                                                                      
    model = None
    solver = None
    input_dims = data_dict['X_train'].shape[1:]
    #1:表示从第一个维度开始，也就是从通道数开始，因为输入的维度是(N,C,H,W)，所以这里的到了(C,H,W)
    model = DeepConvNet(input_dims=input_dims, num_filters=([64]*2)+([128]*2)+([256]*3)+([512]*3)+([512]*3), max_pools=[1,3,6,9,12], batchnorm=True, num_classes=10, weight_scale='kaiming', reg=0.0001, dtype=dtype, device=device)
    #num_filters=[2,2,3,3,3]表示每一层的卷积核的个数，每两个filter为一层，然后2个，然后3个，以此类推，故maxpool为0，1，2，3，4
    solver = Solver(model, data_dict,update_rule=adam, optim_config={'learning_rate': 0.001}, lr_decay=0.95, num_epochs=50, batch_size=128, print_every=100,device=device)
    #adam不加单引号，表示调用的是adam函数，而'adam'表示调用的是字符串'adam'
    return solver

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules.
    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    Example usage might look something like this:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
            update_rule=sgd,
            optim_config={
              'learning_rate': 1e-3,
            },
            lr_decay=0.95,
            num_epochs=10, batch_size=100,
            print_every=100,
            device='cuda')
    solver.train()
    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to torch
      tensors containing parameter values.
    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].
      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].
      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        Optional arguments:
        - update_rule: A function of an update rule. Default is sgd.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - print_acc_every: We will print the accuracy every
          print_acc_every epochs.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", self.sgd)
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.device = kwargs.pop("device", "cpu")

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.print_acc_every = kwargs.pop("print_acc_every", 1)
        self.verbose = kwargs.pop("verbose", True)
        self.adv_ratio = kwargs.pop("adv_ratio", 0.2)
        self.pgd_epsilon = kwargs.pop("pgd_epsilon", 0.031)
        self.pgd_alpha = kwargs.pop("pgd_alpha", 0.008)
        self.pgd_iters = kwargs.pop("pgd_iters", 40)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = torch.randperm(num_train)[: self.batch_size]
        X_batch = self.X_train[batch_mask].to(self.device)
        y_batch = self.y_train[batch_mask].to(self.device)
  

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss.item())

        # Perform a parameter update
        with torch.no_grad():
            for p, w in self.model.params.items():
                dw = grads[p]
                config = self.optim_configs[p]
                next_w, next_config = self.update_rule(w, dw, config)
                self.model.params[p] = next_w
                self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy over Epochs")
        plt.plot(range(1, len(self.train_acc_history) + 1), self.train_acc_history, label='Train')
        plt.plot(range(1, len(self.val_acc_history) + 1), self.val_acc_history, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()   
    

    
    
   
    @staticmethod
    def sgd(w, dw, config=None):
        """
        Performs vanilla stochastic gradient descent.
        config format:
        - learning_rate: Scalar learning rate.
        """
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)

        w -= config["learning_rate"] * dw
        return w, config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = torch.randperm(N, device=self.device)[:num_samples]
            N = num_samples
            X = X[mask]
            y = y[mask]
        X = X.to(self.device)
        y = y.to(self.device)

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(torch.argmax(scores, dim=1))

        y_pred = torch.cat(y_pred)
        acc = (y_pred == y).to(torch.float).mean()

        return acc.item()

    def train(self, time_limit=None, return_best_params=True):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        prev_time = start_time = time.time()

        for t in range(num_iterations):

            cur_time = time.time()
            if (time_limit is not None) and (t > 0):
                next_time = cur_time - prev_time
                if cur_time - start_time + next_time > time_limit:
                    print(
                        "(Time %.2f sec; Iteration %d / %d) loss: %f"
                        % (
                            cur_time - start_time,
                            t,
                            num_iterations,
                            self.loss_history[-1],
                        )
                    )
                    print("End of training; next iteration "
                          "will exceed the time limit.")
                    break
            prev_time = cur_time

            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Time %.2f sec; Iteration %d / %d) loss: %f"
                    % (
                        time.time() - start_time,
                        t + 1,
                        num_iterations,
                        self.loss_history[-1],
                    )
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
               
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            with torch.no_grad():
                first_it = t == 0
                last_it = t == num_iterations - 1
                if first_it or last_it or epoch_end:
                    train_acc = \
                        self.check_accuracy(self.X_train,
                                            self.y_train,
                                            num_samples=self.num_train_samples)
                    val_acc = \
                        self.check_accuracy(self.X_val,
                                            self.y_val,
                                            num_samples=self.num_val_samples)
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    self._save_checkpoint()

                    if self.verbose and self.epoch % self.print_acc_every == 0:
                        print(
                            "(Epoch %d / %d) train acc: %f; val_acc: %f"
                            % (self.epoch, self.num_epochs, train_acc, val_acc)
                        )

                    # Keep track of the best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_params = {}
                        for k, v in self.model.params.items():
                            self.best_params[k] = v.clone()

        # At the end of training swap the best params into the model
        if return_best_params:
            self.model.params = self.best_params
        self.plot_training_history()    



