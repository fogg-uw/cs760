# Question 1

D2z = read.table(file='hw3-1/data/D2z.txt', header=F, col.names = c('x', 'y', 's'))
fenceposts = seq(from = -2, to = 2, by = 0.1)
grid = expand.grid(x = fenceposts, y = fenceposts)
grid$nearest = NA
grid$pred = NA
D2z$dist = 0
for (i in 1:nrow(grid)) {
  for (j in 1:nrow(D2z)) {
    D2z$dist[j] = sqrt((grid$x[i] - D2z$x[j])^2 + (grid$y[i] - D2z$y[j])^2)
  }
  grid$nearest[i] = which(D2z$dist == min(D2z$dist))[1]
  grid$pred[i] = D2z$s[grid$nearest[i]]
}
D2z$dist = NULL
grid$nearest = NULL
alldata = data.frame(
  x = c(D2z$x, grid$x),
  y = c(D2z$y, grid$y),
  s = c(D2z$s, rep(NA, nrow(grid))),
  pred = c(rep(NA, nrow(D2z)), grid$pred)
)
library(ggplot2)
ggplot(data=alldata, aes(x=x, y=y, label=s, color=pred)) + geom_point() + geom_label()

# Question 2

emails = read.csv(file='hw3-1/data/emails.csv', header=T)
t = 0 # set t > 0 to use (1/4)^t of the data (e.g. for fast code testing)
n = 5000/2^t
p = 3000/2^t
emails2 = emails[1:n, c(1:(p+1), 3002)]


my1NN = function(testrange, trainingrange, dataset, p) {
  library(data.table) # base R is slow but data.table is fast
  dataset = data.table(dataset)
  dataset$guess = NA
  y = melt( # molten data is memory-heavy but fast
    data = dataset[trainingrange, 1:(p+1)],
    id.vars = 'Email.No.'
  )
  y = data.table(y)
  library(tictoc) # for timing
  tic()
  for (i in testrange) {
    x = melt(
      data = dataset[i, 1:(p+1)],
      id.vars = 'Email.No.'
    )
    y$value2 = rep(x$value, each = length(trainingrange))
    y$sqdiff = (y$value - y$value2)^2
    dist = y[, .(sum(sqdiff)), by="Email.No."]
    dist$V1 = sqrt(dist$V1)
    nearest = which(dist$V1 == min(dist$V1))[1]
    nearest = trainingrange[nearest]
    #print(paste("dataset$guess[i] = ", dataset$guess[i])) # for debug
    #print(paste("dataset$Prediction[nearest] = ", dataset$Prediction[nearest])) # for debug
    dataset$guess[i] = dataset$Prediction[nearest]
  }
  toc()
  
  accuracy = sum(dataset$Prediction[testrange] == dataset$guess[testrange]) / length(testrange)
  precision = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$guess[testrange])
  recall = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$Prediction[testrange])
  return(c(accuracy, precision, recall))
}

library(class)

testrange = 1:(floor(n/5))
trainingrange = 1:n
trainingrange = trainingrange[!(trainingrange %in% testrange)]
my1NN(testrange, trainingrange, emails2, p)

for(i in 1:5) {
  testrange = seq(from = 1 + (floor(n/5))*(i-1), to = (floor(n/5))*i, by = 1)
  trainingrange = 1:n
  trainingrange = trainingrange[!(trainingrange %in% testrange)]
  print(i)
  print(my1NN(testrange, trainingrange, emails2, p))
  #knn(emails2[testrange,1:(p+1)], emails2[trainingrange,1:(p+1)], as.factor(emails2$Prediction[trainingrange]), k=1)
}

# Question 3

q = 0 # set t > 0 to use (1/4)^t of the data (e.g. for fast code testing)
n = floor(5000/2^q)
p = floor(3000/2^q)
emails2 = emails[1:n, c(1:(p+1), 3002)]
testrange = 1:(floor(n/5))
trainingrange = 1:n
trainingrange = trainingrange[!(trainingrange %in% testrange)]

s = function(z) {
  # takes scalar, vector, or matrix input
  # output has same dimension as input
  y = 1 / (1 + exp(-z))
  return(y)
}

f = function(X, t) {
  # X should be an n x p feature matrix.
  # t should be a p x 1 parameter matrix.
  # ouptut will be an n x 1 prediction matrix.
  y = s( X %*% t )
  return(y)
}

L = function(yhat, y) {
  # yhat and y should both be scalars.
  # output will be a scalar.
  z = -(y * log(yhat) + (1-y)*log(1-yhat))
  return(z)
}

grad = function(X, t, y) {
  # X should be an (n x p) feature matrix.
  # t should be a (p x 1) parameter vector.
  # y should be a (n x 1) predictand vector.
  # output will be a (p x 1) vector.
  
  n = nrow(X)
  
  #print(paste("ncol(t(X)) = ",ncol(t(X))))
  
  a = (y * (1 - f(X, t)))
  b = ((1 - y) * f(X, t))
  a = as.matrix(a, nrow=length(a))
  b = as.matrix(b, nrow=length(b))
  # print(paste("nrow(a) = ", nrow(a)))
  # print(paste("nrow(b) = ", nrow(b)))
  
  g = - t(X) %*% (a - b) / n
  
  return(g)
  
}

myLogReg = function(X, y, e, init, steps) {
  step = 0
  t = init
  while (step < steps) {
    step = step + 1
    g = grad(X, t, y)
    #print(step)
    #print(g)
    t = t - e * g
  }
  return(t)
}


library(tictoc)
for(i in 1:5) {
  print(i)
  testrange = seq(from = 1 + (floor(n/5))*(i-1), to = (floor(n/5))*i, by = 1)
  trainingrange = 1:n
  trainingrange = trainingrange[!(trainingrange %in% testrange)]
  X = as.matrix(emails2[trainingrange, 2:(p+1)], nrow=length(trainingrange))
  X = cbind(rep(1, nrow(X)), X)
  y =           emails2[trainingrange, p+2]
  tic()
  t = myLogReg(
    X = X,
    y = y,
    e = 0.5,
    init = rep(0,p+1),
    steps = 128
  )
  toc()
  X = as.matrix(emails2[testrange, 2:(p+1)], nrow=length(testrange))
  X = cbind(rep(1, nrow(X)), X)
  y =           emails2[testrange, p+2]
  pred = ceiling(s( X %*% t ) - (2/3)) # if prob(spam) > 2/3, classify as spam
  accuracy = sum(pred == y) / length(testrange)
  precision = sum(pred == 1 & y == 1) / sum(pred == 1)
  recall = sum(pred == 1 & y == 1) / sum(y == 1)
  print(c(accuracy, precision, recall))
}



#glmdata = as.data.frame(cbind(X, y))
#colnames(glmdata)[ncol(glmdata)] = 'Prediction'
#t2 = glm(data = glmdata, formula = Prediction ~ ., family = "binomial")

# Question 4

mykNN = function(k, testrange, trainingrange, dataset, p) {
  library(data.table) # base R is slow but data.table is fast
  dataset = data.table(dataset)
  dataset$guess = NA
  y = melt( # molten data is memory-heavy but fast
    data = dataset[trainingrange, 1:(p+1)],
    id.vars = 'Email.No.'
  )
  y = data.table(y)
  library(tictoc) # for timing
  tic()
  for (i in testrange) {
    x = melt(
      data = dataset[i, 1:(p+1)],
      id.vars = 'Email.No.'
    )
    y$value2 = rep(x$value, each = length(trainingrange))
    y$sqdiff = (y$value - y$value2)^2
    dist = y[, .(sum(sqdiff)), by="Email.No."]
    dist$V1 = sqrt(dist$V1)
    dist$itemnum = 1:nrow(dist)
    dist = dist[order(dist$V1),]
    nearest = dist$itemnum[1:k]
    nearest = trainingrange[nearest]
    #print(paste("dataset$guess[i] = ", dataset$guess[i])) # for debug
    #print(paste("dataset$Prediction[nearest] = ", dataset$Prediction[nearest])) # for debug
    predictions = dataset$Prediction[nearest]
    prediction = labels(sort(table(predictions), decreasing = TRUE))[[1]][1]
    prediction = as.numeric(prediction)
    dataset$guess[i] = prediction
  }
  toc()
  
  accuracy = sum(dataset$Prediction[testrange] == dataset$guess[testrange]) / length(testrange)
  precision = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$guess[testrange])
  recall = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$Prediction[testrange])
  return(c(accuracy, precision, recall))
}

testrange = 1:(floor(n/5))
trainingrange = 1:n
trainingrange = trainingrange[!(trainingrange %in% testrange)]

for(k in c(1, 3, 5, 7, 10)) {
  print(paste0("k=", k))
  for(i in 1:5) {
    testrange = seq(from = 1 + (floor(n/5))*(i-1), to = (floor(n/5))*i, by = 1)
    trainingrange = 1:n
    trainingrange = trainingrange[!(trainingrange %in% testrange)]
    print(paste0("i=", i))
    print(mykNN(k, testrange, trainingrange, emails2, p))
    #knn(emails2[testrange,1:(p+1)], emails2[trainingrange,1:(p+1)], as.factor(emails2$Prediction[trainingrange]), k=1)
  }
}

# sorry about what's coming

acc1 = c(0.819, 0.837, 0.836, 0.822, 0.741)
acc3 = c(0.843, 0.840, 0.856, 0.831, 0.746)
acc5 = c(0.837, 0.840, 0.859, 0.756, 0.743)
rm(acc5)
acc5 = c(0.837, 0.840, 0.859, 0.837, 0.750)
acc7 = c(0.839, 0.840, 0.867, 0.840, 0.756)
acc10 = c(0.843, 0.832, 0.877, 0.842, 0.770)

avgAcc = c(
  mean(acc1),
  mean(acc3),
  mean(acc5),
  mean(acc7),
  mean(acc10)
)

plot(x=c(1, 3, 5, 7, 10), y = avgAcc, type='b')



# Question 5

# kNN that reports vote tallies rather than just the winner

mykNN_c = function(k, testrange, trainingrange, dataset, p) {
  library(data.table) # base R is slow but data.table is fast
  dataset = data.table(dataset)
  dataset$guess = NA
  y = melt( # molten data is memory-heavy but fast
    data = dataset[trainingrange, 1:(p+1)],
    id.vars = 'Email.No.'
  )
  y = data.table(y)
  library(tictoc) # for timing
  tic()
  for (i in testrange) {
    x = melt(
      data = dataset[i, 1:(p+1)],
      id.vars = 'Email.No.'
    )
    y$value2 = rep(x$value, each = length(trainingrange))
    y$sqdiff = (y$value - y$value2)^2
    dist = y[, .(sum(sqdiff)), by="Email.No."]
    dist$V1 = sqrt(dist$V1)
    dist$itemnum = 1:nrow(dist)
    dist = dist[order(dist$V1),]
    nearest = dist$itemnum[1:k]
    nearest = trainingrange[nearest]
    #print(paste("dataset$guess[i] = ", dataset$guess[i])) # for debug
    #print(paste("dataset$Prediction[nearest] = ", dataset$Prediction[nearest])) # for debug
    predictions = dataset$Prediction[nearest]
    prediction = mean(predictions)
    dataset$guess[i] = prediction
  }
  toc()
  
  return(dataset$guess[testrange])
  
  #accuracy = sum(dataset$Prediction[testrange] == dataset$guess[testrange]) / length(testrange)
  #precision = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$guess[testrange])
  #recall = sum(dataset$Prediction[testrange] == 1 & dataset$guess[testrange] == 1) / sum(dataset$Prediction[testrange])
  #return(c(accuracy, precision, recall))
}

# ROC algorithm
# let y be 0 or 1 (so that "neg" == 0)

ROC = function(y, c) {
  output1 = numeric()
  output2 = numeric()
  m = length(y)
  if (length(c) != m) {
    errorCondition("y and c have different lengths")
  }
  howtoorder = order(c, decreasing=TRUE)
  c = c[howtoorder]
  y = y[howtoorder]
  num_pos = sum(y)
  num_neg = length(y) - num_pos
  TP = 0
  FP = 0
  last_TP = 0
  for (i in 1:m) {
    # find thresholds where there is a pos instance on the high side,
    # neg instance on the low side
    if (i > 1) {
      if (c[i] != c[i-1] & y[i] == 0 & TP > last_TP) {
        FPR = FP / num_neg
        TPR = TP / num_pos
        output1 = c(output1, FPR)
        output2 = c(output2, TPR)
        last_TP = TP
      }
    }
    if (y[i] == 1) TP = TP + 1
    else FP = FP + 1
  }
  FPR = FP / num_neg
  TPR = TP / num_pos
  output1 = c(output1, FPR)
  output2 = c(output2, TPR)
  output = cbind(output1, output2)
  return(output)
}

testrange = 1:(floor(n/5))
trainingrange = 1:n
trainingrange = trainingrange[!(trainingrange %in% testrange)]
k=5
predictions = mykNN_c(k, testrange, trainingrange, emails2, p)
ROC1 = ROC(emails2$Prediction[testrange], predictions)
X = as.matrix(emails2[trainingrange, 2:(p+1)], nrow=length(trainingrange))
X = cbind(rep(1, nrow(X)), X)
y =           emails2[trainingrange, p+2]
t = myLogReg(
  X = X,
  y = y,
  e = 2^(-6),
  init = rep(0,p+1),
  steps = 2^9
)
X = as.matrix(emails2[testrange, 2:(p+1)], nrow=length(testrange))
X = cbind(rep(1, nrow(X)), X)
y =           emails2[testrange, p+2]
pred = s( X %*% t )
ROC2 = ROC(emails2$Prediction[testrange], pred)

q5graphdata = rbind(
  cbind(ROC1, rep(1, nrow(ROC1))),
  cbind(ROC2, rep(2, nrow(ROC2)))
)
colnames(q5graphdata) = c('FPR', 'TPR', 'algorithm')
q5graphdata = as.data.frame(q5graphdata)
q5graphdata$algorithm = as.factor(q5graphdata$algorithm)
ggplot(data=q5graphdata, aes(
  x = FPR,
  y = TPR,
  linetype = algorithm,
  color = algorithm
)) + geom_line() + geom_point()

