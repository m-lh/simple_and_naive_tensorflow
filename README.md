# simple_and_naive_tensorflow
反向传播算法的简单实现，对照了tf的api

- 只支持标量计算
- 只实现了几个运算符

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
