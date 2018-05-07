# coding=utf-8
#!/usr/bin/python
#Training data set
#each element in x represents (x0,x1,x2)
import random
x = [(1,0.,3) , (1,1.,3) ,(1,2.,3), (1,3.,2) , (1,4.,4)]
#y[i] is the output of y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
y = [95.364,97.217205,75.195834,60.105519,49.342380]


epsilon = 0.0001
#learning rate
alpha = 0.01
diff = [0,0]
error1 = 0
error0 =0
m = len(x)


#init the parameters to zero
theta0 = 0
theta1 = 0
theta2 = 0

epoch = 0
while True:

    #calculate the parameters
    # 线性回归：h(x) = theta0 + theta1 * x[i][1] + theta2 * x[i][2]  
    # 损失函数：(1/2) *  (y - h(x)) ^ 2
    # theta0 = theta0 - (  - alpha * (y - h(x))* 1 )
    # theta1 = theta1 - (  - alpha * (y - h(x))* x[i][1] )
    # theta2 = theta2 - (  - alpha * (y - h(x))* x[i][2] )
    # 1. 随机梯度下降算法在迭代的时候，每迭代一个新的样本，就会更新一次所有的theta参数。
    i = random.randint(0, m - 1)
        
    # (y - h(x))
    diff[0] = y[i]-( theta0 + theta1 * x[i][1] + theta2 * x[i][2] )
    # - (y - h(x))x
    gradient0 = - diff[0]* x[i][0]
    gradient1 = - diff[0]* x[i][1]
    gradient2 = - diff[0]* x[i][2]
    # theta = theta - (  - alpha * (y - h(x))x )
    theta0 = theta0 - alpha * gradient0
    theta1 = theta1 - alpha * gradient1
    theta2 = theta2 - alpha * gradient2
    
    #calculate the cost function
    error1 = 0
    # 此处error为一个相对的Error值。
    for i in range(len(x)):
        error1 += ( y[i]-( theta0 + theta1 * x[i][1] + theta2 * x[i][2] ) )**2/2
    
    if abs(error1-error0) < epsilon:
        break
    else:
        error0 = error1
    epoch += 1
    print(' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f, epoch : % f'%(theta0,theta1,theta2,error1, epoch))

print('Done: theta0 : %f, theta1 : %f, theta2 : %f'%(theta0,theta1,theta2))