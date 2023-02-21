import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20230217)

def generate_x(n):
    x = rng.choice([0,1], size=1, p=[0.5,0.5])
    tmp_x = x
    for _ in range(n-1):
        if tmp_x == 0:
            tmp_x = rng.choice([0,1], size=1, p=[0.99,0.01])
        else:
            tmp_x = rng.choice([0,1], size=1, p=[0.03,0.97])
        x = np.append(x, tmp_x)
    return x

def generate_y(x, sigma):
    return rng.normal(x, sigma)

def optim_x_map(y, n, sigma):
    patterns = np.array([[0,0], [0,1], [1,0], [1,1]])
    prob_products = np.zeros(4)
    
    for i, pattern in enumerate(patterns):
        tmp_product = 1
        tmp_product *= get_prob_x2x(pattern[0], pattern[1])
        tmp_product *= get_prob_x2y(pattern[0], y[0], sigma)
        tmp_product *= get_prob_x2y(pattern[1], y[1], sigma)
        prob_products[i] = tmp_product
    
    
    if prob_products[0] > prob_products[2]:
        xmap1 = patterns[0]
        prob_products[2] = prob_products[0]
    else:
        xmap1 = patterns[2]
        prob_products[0] = prob_products[2]
    
    if prob_products[1] > prob_products[3]:
        xmap2 = patterns[1]
        prob_products[3] = prob_products[1]
    else:
        xmap2 = patterns[3]
        prob_products[1] = prob_products[3]
    
    xmaps = np.array([xmap1, xmap2])
    
    patterns = np.array([[0,0], [1,0], [0,1], [1,1]])

    for i in range(2, n-1):
        for j, pattern in enumerate(patterns):
            tmp_product = 1
            tmp_product *= get_prob_x2x(pattern[0], pattern[1])
            tmp_product *= get_prob_x2y(pattern[1], y[i], sigma)
            prob_products[j] *= tmp_product
        
        if prob_products[0] > prob_products[1]:
            xmap1 = np.append(xmaps[0], 0)
            idx1 = 0
        else:
            xmap1 = np.append(xmaps[1], 0)
            idx1 = 1
        
        if prob_products[2] > prob_products[3]:
            xmap2 = np.append(xmaps[0], 1)
            idx2 = 2
        else:
            xmap2 = np.append(xmaps[1], 1)
            idx2 = 3
        
        if idx1 == 0:
            prob_products[1] = prob_products[idx2]
        else:
            prob_products[0] = prob_products[idx2]
        
        if idx2 == 2:
            prob_products[3] = prob_products[idx1]
        else:
            prob_products[2] = prob_products[idx1]
        
        xmaps = np.array([xmap1, xmap2])
    
    if prob_products[0] > prob_products[1]:
        return xmaps[0]
    else:
        return xmaps[1]
    
def get_prob_x2x(x1, x2):
    """
    x1 : x_i
    x2 : x_(i+1)
    """
    if x1 == 0:
        if x2 == 0:
            return 0.99
        else:
            return 0.01
    else:
        if x2 == 0:
            return 0.03
        else:
            return 0.97
    
def get_prob_x2y(x, y, sigma):
    return 1 / (np.sqrt(2*np.pi)*sigma) * np.exp(-(y-x)*(y-x)/2*sigma*sigma)
    
def q1(n=200):
    plt.figure()
    for i in range(10):
        x = generate_x(n)
        plt.plot(x+3*i)
    plt.show()
    
def q2(n=200, sigma=0.7):
    x = generate_x(n)
    y = generate_y(x, sigma)
    plt.figure()
    plt.plot(x)
    plt.plot(y, marker='.', ls='None')
    plt.show()

def q4(n=200, sigma=0.7):
    x = generate_x(n)
    y = generate_y(x, sigma)
    x_map = optim_x_map(y, n, sigma)
    plt.figure()
    plt.plot(x)
    plt.plot(x_map)
    plt.show()

if __name__ == "__main__":
    # q1()
    # q2()
    q4()
