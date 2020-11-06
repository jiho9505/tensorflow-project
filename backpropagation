import tensorflow as tf
import numpy as np

tf.set_random_seed(100)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([784, 300]))
b1 = tf.Variable(tf.random_normal([300]))
w2 = tf.Variable(tf.random_normal([300, 30]))
b2 = tf.Variable(tf.random_normal([30]))
w3 = tf.Variable(tf.random_normal([30, 10]))
b3 = tf.Variable(tf.random_normal([10]))
## b1,b2,b3에 [1,300],[1,30],[1,10] 해줘도 상관 없음.

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(-x)))
## sigmoid의 식을 풀어 쓴 것.

def sigma_prime(x):
    return sigma(x) * (1 - sigma(x))

## sigmoid의 역전파(sigma를 미분한 값)

def softm(x):
    exps = tf.exp(x)
    return exps / np.sum(exps)

## softmax의 식을 풀어 쓴 것.

l1 = tf.add(tf.matmul(X, w1), b1)
a1 = sigma(l1)
l2 = tf.add(tf.matmul(a1, w2), b2)
a2 = sigma(l2)
l3 = tf.add(tf.matmul(a2, w3), b3)
y_pred = tf.nn.softmax(l3)
##y_pred는 우리가 아는 hypothesis
##softm 함수가 구현이 안되서 일단 역전파를 통해서 accuracy를 높이는것이 목표이므로 텐서의 softmax를 사용.
## 입력과 출력 사이에 3개의 layer가 존재. l1에서 l2갈 때 activation 함수는 sigmoid 이런식으로 이해.

assert y_pred.shape.as_list() == Y.shape.as_list() ## 예측값과 정답값의 shape이 같지 않으면 error 출력
diff = (y_pred - Y) ## diff는 오차
## 일단 diff가 소프트맥스를 사용하면 local gradient 가 되는 듯.
## 이 밑으로는 역전파 과정 (weight를 update 시켜주는 것. 마치 optimizer를 돌려서 cost 값을 낮춰주듯이)
d_l3 =  diff
d_b3 = d_l3
d_w3 = tf.matmul(tf.transpose(a2), d_l3) ## d_l3와 matmul 행렬곱을 할수 있도록 a2를 치환
## diff가 softmax의 역전파로 추정. 아니 일단 역전파라고 할순없음. 필요한 값이 저게 맞긴 한데 솦맥의 역전파라곤 할수없음.
## 왜 역으로 가는데 a2를 곱한걸까?

d_a2 = tf.matmul(d_l3, tf.transpose(w3))
d_l2 = d_a2 * sigma_prime(l2)
d_b2 = d_l2
d_w2 = tf.matmul(tf.transpose(a1), d_l2)

d_a1 = tf.matmul(d_l2, tf.transpose(w2))
d_l1 = d_a1 * sigma_prime(l1)
d_b1 = d_l1
d_w1 = tf.matmul(tf.transpose(X), d_l1)

learning_rate = 0.1

step = [
    tf.assign(w1, w1 - learning_rate * d_w1),
    tf.assign(b1, b1 - learning_rate *
              tf.reduce_mean(d_b1, reduction_indices=[0])),
    tf.assign(w2, w2 - learning_rate * d_w2),
    tf.assign(b2, b2 - learning_rate *
              tf.reduce_mean(d_b2, reduction_indices=[0])),
    tf.assign(w3, w3 - learning_rate * d_w3),
    tf.assign(b3, b3 - learning_rate *
              tf.reduce_mean(d_b3, reduction_indices=[0]))
]
## reduction_indices 0을 전달하면 열 합계 1을 전달하면 행 합계 아무것도 전달하지 않으면 전체 합
## assign이 새로운 값을 할당하는 것. 예를 들어 assign(a,b)이면 b의 값을 a로 다시 할당. a는 결국 b가 된다.
## 저 step 과정이 weight 값들을 올바르게 다시 설정해주고 있음.

is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        sess.run(step, feed_dict={X: batch_xs,Y: batch_ys})
        if i % 1000 == 0:  
            answer = sess.run(accuracy, feed_dict={X: mnist.test.images[:1000],Y: mnist.test.labels[:1000]})
            print(answer)
            ## [:1000]은 0-999까지 test.images 개수.
