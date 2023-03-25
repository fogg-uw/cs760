import os
import numpy as np
from functools import reduce
from operator import itemgetter
import random

print("hello world")

# Question 2

files = os.listdir("hw4-1/languageID")
files = sorted(files)
N = len(files)
labels = [" "] * N
bags = []

for i in range(N):
    bag = []
    labels[i] = files[i][0]
    with open("hw4-1/languageID/" + files[i]) as f:
        lines = f.readlines()
        for line in lines:
            for char in line:
                if char != '\n':
                    bag.append(char)
    bags.append(bag)


# Question 2.1

print("\nQuestion 2.1\n")

training = []
for i in range(0,10): training.append(i)
for i in range(20,30): training.append(i)
for i in range(40,50): training.append(i)
N_tr = len(training)

alpha = 1/2
label_count = np.array([0,0,0]) # en, jp, sp
langs = np.array(["e", "j", "s"])
langdict = {
    "e": 0,
    "j": 1,
    "s": 2
}
L = len(label_count)
for label in [labels[i] for i in training]:
    label_count[langdict[label]] += 1

prior = np.array([1/3, 1/3, 1/3]) # initialize to equal priors

for i in range(L):
    prior[i] = (label_count[i] + alpha) / (N_tr + L*alpha)

print(prior)

# Question 2.2

print("\nQuestion 2.2\n")

ccpe = [] # class conditional probability for english
letterdict = {
    "a":0,
    "b":1,
    "c":2,
    "d":3,
    "e":4,
    "f":5,
    "g":6,
    "h":7,
    "i":8,
    "j":9,
    "k":10,
    "l":11,
    "m":12,
    "n":13,
    "o":14,
    "p":15,
    "q":16,
    "r":17,
    "s":18,
    "t":19,
    "u":20,
    "v":21,
    "w":22,
    "x":23,
    "y":24,
    "z":25,
    " ":26
}
KS = 27
for k in range(KS):
    ccpe.append(0)
for i in range(N_tr):
    if labels[i] == "e":
        for j in range(len(bags[training[i]])):
            ccpe[letterdict[bags[training[i]][j]]] += 1

denom = 0
for k in range(27):
    ccpe[k] = ccpe[k] + alpha
    denom = denom + ccpe[k]
for k in range(27):
    ccpe[k] = ccpe[k] / denom

print(ccpe)

# Question 2.3

print("\nQuestion 2.3\n")

ccp = []
for l in range(3):
    ccpl = []
    for k in range(KS):
        ccpl.append(0)
    for i in range(N_tr):
        if labels[training[i]] == langs[l]:
            for j in range(len(bags[training[i]])):
                ccpl[letterdict[bags[training[i]][j]]] += 1

    denom = 0
    for k in range(27):
        ccpl[k] = ccpl[k] + alpha
        denom = denom + ccpl[k]
    for k in range(27):
        ccpl[k] = ccpl[k] / denom

    ccp.append(ccpl)

print(ccp[1])
print(ccp[2])

# Question 2.4

print("\nQuestion 2.4\n")

test = 11
counts = [0]*27
for word in bags[test]:
    counts[letterdict[word]] += 1
print(counts)

# Question 2.5

print("\nQuestion 2.5\n")

prob_vec = []
for y in range(3):
    logprob = 0
    for x in range(27):
        logprob += counts[x] * np.log(ccp[y][x])
    prob_vec.append(logprob)
print(prob_vec)

# Question 2.6

print("\nQuestion 2.6\n")

# using bayes rule, it's equivalent if we subtract off
# the smallest of the log probs from everything

# priors are also the same for everything so we can ignore them.

min_prob = min(prob_vec)
posterior = prob_vec
for i in range(len(prob_vec)):
    prob_vec[i] = prob_vec[i] - min_prob
    posterior[i] = np.exp(prob_vec[i])

evidence = np.sum(posterior)
for i in range(len(posterior)):
    posterior[i] = posterior[i] / evidence

print(posterior)
MAP_val = max(posterior)
MAP_idx = posterior.index(MAP_val)
print("best label estimate is " + langs[MAP_idx] + " with probability " + str(MAP_val))

# Question 2.7

print("\nQuestion 2.7\n")

tests = []
for i in range(10,20): tests.append(i)
for i in range(30,40): tests.append(i)
for i in range(50,60): tests.append(i)
N_te = len(tests)

confusion = np.array([[0,0,0],[0,0,0],[0,0,0]])

for test in tests:
    counts = [0]*27
    for word in bags[test]:
        counts[letterdict[word]] += 1

    prob_vec = []
    for y in range(3):
        logprob = 0
        for x in range(27):
            logprob += counts[x] * np.log(ccp[y][x])
        prob_vec.append(logprob)

    min_prob = min(prob_vec)
    likelihood = prob_vec
    for i in range(len(prob_vec)):
        prob_vec[i] = prob_vec[i] - min_prob
        likelihood[i] = np.exp(prob_vec[i])

    evidence = np.sum(likelihood)
    posterior = likelihood
    for i in range(len(posterior)):
        if likelihood[i] == np.inf: posterior[i] = 1
        else: posterior[i] = likelihood[i] / evidence



    MAP_val = max(posterior)
    MAP_idx = posterior.index(MAP_val)

    true_idx = langdict[labels[test]]

    confusion[MAP_idx][true_idx] += 1

print(confusion)

# Question 2.8

print("\nQuestion 2.8\n")

test = 11

# my algorithm: take two random letters and switch them.  do this many times.

manytimes = 1000
random.seed(1)
for time in range(manytimes):
    rand1 = random.randrange(len(bags[test]))
    rand2 = random.randrange(len(bags[test]))
    val1 = bags[test][rand1]
    val2 = bags[test][rand2]
    bags[test][rand1] = val2
    bags[test][rand2] = val1

counts = [0]*27
for word in bags[test]:
    counts[letterdict[word]] += 1

prob_vec = []
for y in range(3):
    logprob = 0
    for x in range(27):
        logprob += counts[x] * np.log(ccp[y][x])
    prob_vec.append(logprob)


min_prob = min(prob_vec)
likelihood = prob_vec
for i in range(len(prob_vec)):
    prob_vec[i] = prob_vec[i] - min_prob
    likelihood[i] = np.exp(prob_vec[i])

evidence = np.sum(likelihood)
posterior = likelihood
for i in range(len(posterior)):
    if likelihood[i] == np.inf: posterior[i] = 1
    else: posterior[i] = likelihood[i] / evidence


print(posterior)
MAP_val = max(posterior)
MAP_idx = posterior.index(MAP_val)
print("best label estimate is " + langs[MAP_idx] + " with probability " + str(MAP_val))

# Question 3

def sigma(x):
    y = 1 / (1 + np.exp(x))
    return(y)

def softmax(x):
    y = np.exp(x) / (np.sum(np.exp(x)))
    return(y)

def dLdg(y, g):
    z = -y/g
    return(z)

def dgdsx(g):
    k = len(g)
    d = np.array()
    for i in range(k):
        for j in range(k):
            d[i][j] = g[i] * ((i == j) - g[j])
    return(d)

def propforward(W1, W2, W3, x):
    a1 = sigma(np.matmul(W1, x))
    a2 = sigma(np.matmul(W2, a1))
    yhat = softmax(np.matmul(W3, a2))
    out = np.array([a1, a2, yhat])
    return(out)

def propback(W1, W2, W3, x, a1, a2, yhat, y, alpha):
    k = len(y)
    d2 = W3.shape[1]
    d1 = W2.shape[1]
    d = W1.shape[1]
    gW3 = np.array()
    for p in range(k):
        for q in range(d2):
            si = 0
            for i in range(k):
                sj = 0
                for j in range(k):
                    sj += yhat[i]*((i==j) - yhat[j]) * (p==h) * a2[q]
                si += -(y[i]/yhat[i]) * sj
            gW3[p][q] = si
    gW2 = np.array()
    for p in range(d2):
        for q in range(d1):
            si = 0
            for i in range(k):
                sj = 0
                for j in range(k):
                    sl = 0
                    for l in range(d2):
                        sl += W3[j][l] * sigma(np.matmul(W2,a1)) * (1 - sigma(np.matmul(W2,a1))) * (p==l) * a1[q]
                    sj += yhat[i]*((i==j) - yhat[j]) * sl
                si += -(y[i]/yhat[i]) * sj
            gW2[p][q] = si
    gW1 = np.array()
    for p in range(d1):
        for q in range(d):
            si = 0
            for i in range(k):
                sj = 0
                for j in range(k):
                    sl = 0
                    for l in range(d2):
                        sr = 0 
                        for r in range(d1):
                            sr += W1[l][r] * sigma(np.matmul(W1,x)) * (1 - sigma(np.matmul(W1,x))) * (p==r) * x[q]
                        sl += W3[j][l] * sigma(np.matmul(W2,a1)) * (1 - sigma(np.matmul(W2,a1))) * sr
                    sj += yhat[i]*((i==j) - yhat[j]) * sl
                si += -(y[i]/yhat[i]) * sj
            gW1[p][q] = si
    W1 = W1 - alpha * gW1
    W2 = W2 - alpha * gW2
    W3 = W3 - alpha * gW3
    out = np.array([W1, W2, W3])
    return(out)





