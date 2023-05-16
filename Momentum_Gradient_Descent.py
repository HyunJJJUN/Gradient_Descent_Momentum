import numpy as np
import matplotlib.pyplot as plt

def momentum_gradient_descent(gradient_func, init_theta, learning_rate, gamma, num_iterations):
    theta = init_theta
    v = np.zeros_like(init_theta)
    
    theta_history = []
    for i in range(num_iterations):
        gradient = gradient_func(theta)
        v = gamma*v + learning_rate*gradient
        theta = theta - v
        theta_history.append(theta)
        
    return theta_history

def f(x):
    return x**2 + 10*np.sin(x)

def f_derivative(x):
    return 2*x + 10*np.cos(x)

x = np.linspace(-10, 10, 100)
y = f(x)

init_theta = -9
learning_rate = 0.1
gamma = 0.9
num_iterations = 50

theta_history = momentum_gradient_descent(f_derivative, init_theta, learning_rate, gamma, num_iterations)

plt.plot(x, y)
plt.plot(theta_history, f(np.array(theta_history)), 'ro')
plt.title('Momentum Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
