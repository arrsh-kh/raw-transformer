import random
import math

class Tensor:

    def backward(self):
        if len(self.data) != 1:
            raise ValueError("len(self.data) != 1, backward()")
        if self.requires_grad == False:
            return

        topo = []

        visited = []

        def dfs(t):
            if t in visited:
                return
            visited.append(t)
            for parent in t._prev:
                dfs(parent)
            topo.append(t)

        dfs(self)

        self.grad[0] = 1.0

        for t in topo[::-1]:
            t._backward()

    def garbage_float_array(shape):   # I know making this is kinda useless but the garbage collector can deal with it
        if len(shape) == 0:
            return random.uniform(-1e-2, 1e-2)

        size = shape[0]
        rest = shape[1:]
        return [Tensor.garbage_float_array(rest) for _ in range(size)]

    def flatten(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(Tensor.flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def making_strides(shape):
        stride = ()
        mult = 1
        for i in range(len(shape)-2, -1, -1):
            mult *= shape[i+1]
            stride += (mult,)
        stride = stride[::-1] + (1,)
        return stride

    def offset(self, *indices):
        s = 0
        for i in range(len(self.shape)):
            s += indices[i]*self.strides[i]
        return s

    def get(self, *indices):
        index = self.offset(*indices)
        return self.data[index]

    def set(self, *indices, value):
        index = self.offset(*indices)
        self.data[index] = value

    def __init__(self, shape):
        self.data = Tensor.flatten(Tensor.garbage_float_array(shape))
        self.shape = shape
        self.strides = Tensor.making_strides(self.shape)
        self.requires_grad = False
        self.grad = [0.0] * len(self.data)
        self._prev = []
        self._backward = lambda: None

def add_tensors(t1, t2):
    if t1.shape != t2.shape:
        raise ValueError("Size must be same for both tensors")

    t3 = Tensor(t1.shape)
    for i in range(len(t1.data)):
        t3.data[i] = t1.data[i] + t2.data[i]
    t3._prev = [t1, t2]
    if t1.requires_grad or t2.requires_grad:
        t3.requires_grad = True

    def _backward():
        if t1.requires_grad:
            for i in range(len(t3.data)):
                t1.grad[i] += t3.grad[i] * 1
        if t2.requires_grad:
            for i in range(len(t3.data)):
                t2.grad[i] += t3.grad[i] * 1

    t3._backward = _backward
    return t3


def sub_tensors(t1, t2):
    if t1.shape != t2.shape:
        raise ValueError("Size must be same for both tensors")
    t3 = Tensor(t1.shape)
    for i in range(len(t1.data)):
        t3.data[i] = t1.data[i] - t2.data[i]
    t3._prev = [t1, t2]
    if t1.requires_grad or t2.requires_grad:
        t3.requires_grad = True

    def _backward():
        if t1.requires_grad:
            for i in range(len(t3.data)):
                t1.grad[i] += t3.grad[i] * 1
        if t2.requires_grad:
            for i in range(len(t3.data)):
                t2.grad[i] += t3.grad[i] * -1

    t3._backward = _backward
    return t3

def mul_tensors(t1, t2):
    if t1.shape != t2.shape:
        raise ValueError("Size must be same for both tensors")
    t3 = Tensor(t1.shape)
    for i in range(len(t1.data)):
        t3.data[i] = t1.data[i] * t2.data[i]
    t3._prev = [t1, t2]

    if t1.requires_grad or t2.requires_grad:
        t3.requires_grad = True

    def _backward():
        if t1.requires_grad:
            for i in range(len(t3.data)):
                t1.grad[i] += t3.grad[i] * t2.data[i]
        if t2.requires_grad:
            for i in range(len(t3.data)):
                t2.grad[i] += t3.grad[i] * t1.data[i]

    t3._backward = _backward
    return t3

def add_scalar_tensor(t1, s1):
    t2 = Tensor(t1.shape)
    for i in range(len(t1.data)):
        t2.data[i] = t1.data[i] + s1 
    return t2

def mul_scalar_tensor(t1, s1):
    t2 = Tensor(t1.shape)
    for i in range(len(t1.data)):
        t2.data[i] = t1.data[i] * s1 
    return t2

def matmul(t1, t2):
    m = t1.shape[0]
    k = t1.shape[1]
    n = t2.shape[1]
    assert t2.shape[0] == k
    t3 = Tensor((m, n))

    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += t1.get(i, p) * t2.get(p, j)
            t3.set(i, j, value=acc)

    return t3

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = Tensor((in_features, out_features))
        self.b = Tensor((out_features,))

    def forward(self, x):
        batch = x.shape[0]
        assert x.shape[1] == self.in_features

        y = matmul(x, self.W)

        for i in range(batch):
            for j in range(self.out_features):
                y.set(i, j, value=y.get(i, j) + self.b.get(j))

        return y

def transpose2d(t):
    rows = t.shape[0]
    columns = t.shape[1]

    out = Tensor((columns, rows))

    for i in range(rows):
        for j in range(columns):
            out.set(j, i, value=t.get(i, j))
    return out

def softmax_rows(t):
    rows = t.shape[0]
    columns = t.shape[1]
    out = Tensor(t.shape)
    rowmax = []
    for i in range(rows):
        max = -1e12
        for j in range(columns):
            value = t.get(i, j)
            if max<value:
                max = value
        rowmax.append(max)

    for i in range(rows):
        sum = 0
        for j in range(columns):
            out.set(i, j, value=math.exp(t.get(i, j) - rowmax[i]))
            sum += out.get(i, j)
        for j in range(columns):
            out.set(i, j, value=out.get(i, j)/sum)

    return out

class SelfAttention:

    def __init__(self, d_model):
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.scale = 1/math.sqrt(d_model)

    def forward(self, X):
        Q = self.Wq.forward(X)
        K = self.Wk.forward(X)
        V = self.Wv.forward(X)

        K_T = transpose2d(K)

        scores = matmul(Q, K_T)
        scores = mul_scalar_tensor(scores, self.scale)

        weights = softmax_rows(scores)

        out = matmul(weights, V)

        return out

def relu(t):
    t2 = Tensor(t.shape)
    for i in range(len(t.data)):
        if t.data[i] < 0:
            t2.data[i] = 0
        else:
            t2.data[i] = t.data[i]
    t2._prev = [t]
    t2.requires_grad = t.requires_grad

    def _backward():
        for i in range(len(t.data)):
            if t.data[i] > 0:
                t.grad[i] += t2.grad[i]

    t2._backward = _backward
    return t2

class FeedForward:
    def __init__(self, d_model, d_hidden):
        self.lin1 = Linear(d_model, d_hidden)
        self.lin2 = Linear(d_hidden, d_model)

    def forward(self, X):
        out1 = self.lin1.forward(X)
        out1_act = relu(out1)
        out2 = self.lin2.forward(out1_act)
        return out2

class LayerNorm:
    def __init__(self, d_norm):
        self.d_norm = d_norm
        self.gamma = Tensor((d_norm,))
        self.gamma.data = [1.0]*len(self.gamma.data)
        self.beta = Tensor((d_norm,))
        self.beta.data = [0.0]*len(self.gamma.data)
        self.epsilon = 1e-5

    def forward(self, X):
        batch = X.shape[0]
        d_model = X.shape[1]

        if d_model != self.d_norm:
            raise ValueError("d_model not equals to self.d_norm")

        out = Tensor(X.shape)

        row_vars = [0]*batch
        row_means = [0]*batch
        mean = 0

        for i in range(batch):
            acc = 0.0
            for j in range(d_model):
                acc += X.get(i, j)
            row_means[i] = acc/d_model

        for i in range(batch):
            acc = 0.0
            mean = row_means[i]
            for j in range(d_model):
                x = X.get(i, j)
                diff = x - mean
                acc += diff**2
            row_vars[i] = acc/d_model

        for i in range(batch):
            mean = row_means[i]
            var = row_vars[i]
            std = math.sqrt(var+self.epsilon)
            for j in range(d_model):
                x = X.get(i, j)
                x_hat = (x - mean)/std
                g = self.gamma.get(j)
                b = self.beta.get(j)
                y = g * x_hat + b
                out.set(i, j, value=y)

        return out

class TransformerBlock:
    def __init__(self, d_model, d_hidden):
        self.attn = SelfAttention(d_model)
        self.ffn = FeedForward(d_model, d_hidden)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, X):
        X_norm = self.ln1.forward(X)
        attn_out = self.attn.forward(X_norm)
        A = add_tensors(X, attn_out)
        A_norm = self.ln2.forward(A)
        ffn_out = self.ffn.forward(A_norm)
        Y = add_tensors(A, ffn_out)
        return Y

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = Tensor((vocab_size, d_model))

    def forward(self, tokens): # tokens is a list of ints
        seq_len = len(tokens)
        out = Tensor((seq_len, self.d_model))

        for i in range(seq_len):
            token_id = tokens[i]
            for j in range(self.d_model):
                val = self.weight.get(token_id, j)
                out.set(i, j, value=val)
        return out

class PositionalEmbedding:
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.pos_weight = Tensor((max_len, d_model))

    def forward(self, X):
        seq_len = X.shape[0]
        d_model = X.shape[1]
        if self.max_len < seq_len:
            raise ValueError("seq_len more than max_len")
        for i in range(seq_len):
            for j in range(d_model):
                X.set(i, j, value=(self.pos_weight.get(i, j)+X.get(i, j)))
        return X

class TransformerEncoder:
    def __init__(self, d_model, d_hidden, num_layers):
        self.layers = [TransformerBlock(d_model, d_hidden) for _ in range(num_layers)]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

class Model:
    def __init__(self, d_model, d_hidden, vocab_size, max_len, num_layers):
        self.tok_embed = Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len, d_model)
        self.encoder = TransformerEncoder(d_model, d_hidden, num_layers)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, X):
        X = self.tok_embed.forward(X)
        X = self.pos_embed.forward(X)
        X = self.encoder.forward(X)
        logits = self.lm_head.forward(X)
        return logits
