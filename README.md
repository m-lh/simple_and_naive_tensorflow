# 反向传播算法的简单实现，对照了tf的api

最近在读《Deep Learnning》6.5节有反向传播算法的伪代码实现，于是手有点痒，自己拿python简单实现了一下。

- 只支持标量计算
- 只实现了几个运算符
- 只在Python 3.5.2 win32测试

test.py
```python
import simple_and_naive_tensorflow as tf
import numpy as np

# Prepare train data
# w=2, b=10, err=0.33
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder("float", "X")
Y = tf.placeholder("float", "Y")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X * w - b)
train_op = tf.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x, Y: y})
        print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
        epoch += 1
```
运行结果：
```
Epoch: 1, w: 0.4595379820099185, b: 10.531095859048182
Epoch: 2, w: 0.848309733575573, b: 10.28315942068218
Epoch: 3, w: 1.1627324038410223, b: 10.138667736120407
Epoch: 4, w: 1.3998599121673063, b: 10.073337093187133
Epoch: 5, w: 1.5729436606608966, b: 10.045835603782285
Epoch: 6, w: 1.6977854906747174, b: 10.03272031502392
Epoch: 7, w: 1.7873942222185277, b: 10.024439845827198
Epoch: 8, w: 1.8515331075874357, b: 10.01767370977966
Epoch: 9, w: 1.897331959791685, b: 10.011549642886184
Epoch: 10, w: 1.9299549149394077, b: 10.00601624181811
```

## 简单使用
```
X = placeholder("float", 'X')
w = Variable(77.0, name="weight")
c = Constant(2)  # 常量必需使用Constant
z=X*w*c
print(z.eval({X:2}))
pprint(z)
```

## 具体实现
（有待于完善）
首先定义计算图的节点。
```
class Node:
    def __init__(self, *args):
        self.args = args

        self.children = set()
        self.parents = set()
        
        _graph.add(self)
    
    def eval(self, feed_dict=None):
        raise NotImplementedError()

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    ······

```
节点是实现的核心，拥有各种运算，当然运算返回的还是节点。

这里的add是AddOperation节点的包装
```
def add(op1, op2):
    return AddOperation('add', op1, op2)
```

然后节点有四种，

- 根节点：
  - Variable：变量：可以进行AssignOperation改变值
  - Constant：常量：值不会变化
  - PlaceHolder：占位符：feed_value才会有意义
- 中间节点：
  - Operation：操作：各种操作，并且实现bprop（反向传播）方法

每个节点的eval方法是在session的run是调用，真正传递数值进行计算

```python
class Variable(Node):
    def eval(self, feed_dict=None):
        return self.value

class Constant(Node):
    def eval(self, feed_dict=None):
        return self.value

class PlaceHolder(Node):
    def eval(self, feed_dict=None):
        return feed_dict[self]

class Operation(Node):
    def eval(self, feed_dict=None):
        raise NotImplementedError()
    def bprop(self, *a):
        raise NotImplementedError()

```

对于加法操作的实现
```
class AddOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)+self.args[1].eval(feed_dict)

    def bprop(self, I, V, D):
        return D
        
    def __repr__(self):
        return "%s + %s"%self.args
```

然后是优化器
```
class GradientDescentOptimizer:
    def __init__(self, lr):
        self.lr = Constant(lr)
    def minimize(self, z):
        varibles = list(_varibles)
        g = make_grad_table(varibles, _graph, z)
        _op_value = [k - self.lr * g[k] for k in varibles]
        return AssignOperation("assign", varibles, _op_value)
```

session的简单实现
```
class Session:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, T, feed_dict=None):
        return [V.eval(feed_dict) for V in T]
```
2017年5月24日 花了两天时间
