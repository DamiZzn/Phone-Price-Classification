library(tidyverse)
library(ggplot2)
library(GGally)
library(readr)
library(caTools)
library(class)
library(randomForest)
library(Hmisc)
library(corrplot)
library(caret)
library(gridExtra)
library(RColorBrewer) 
library(mlbench)




### load data
data = read.csv("train.csv")
testData = read.csv("test.csv")
attach(data)
attach(testData)

### exploring data
summary(data)
str(data)
View(data)
describe(data)

# checking for missing values
sum(is.na(data))

# Checking for the unique values
unique(data$price_range)

# correlation between variables
numeric_var <- sapply(data, is.numeric)
corr_matrix <- cor(data[,numeric_var])
corrplot(corr_matrix, main="\n\nCorrelation Plot for Numerical Variables", method="color")



### VISUALIZATION 
# Price Distribution
ggplot(data, aes(x=data[,21])) + 
  geom_histogram(binwidth=1, color="white", fill="lightpink3") +
  labs(x=colnames(data)[21], y= "distribution of prices" ) 

## converting numeric data to factor
data[data$price_range == 0,]$price_range <- "très bas prix"
data[data$price_range == 1,]$price_range <- "bas prix"
data[data$price_range == 2,]$price_range <- "prix moyen"
data[data$price_range == 3,]$price_range <- "prix élevé"
data$price_range <- as.factor(data$price_range)
data$dual_sim <- as.factor(data$dual_sim)
data$four_g <- as.factor(data$four_g)
data$three_g <- as.factor(data$three_g)
data$wifi <- as.factor(data$wifi)
data$blue <- as.factor(data$blue)
data$touch_screen <- as.factor(data$touch_screen)

str(data)

# BOXPLOT
p1<-ggplot(data, aes(x=price_range, y = battery_power, color=price_range)) +
  geom_boxplot() +
  labs(title = "Battery Power vs Price Range")
p2<- ggplot(data, aes(x=price_range, y = ram, color=price_range)) +
  geom_boxplot() +
  labs(title = "RAM vs Price Range")
grid.arrange(p1,p2,nrow=1)


# DISTRIBUTION OF CAMERA QUALITY
cam = data.frame(MegaPixels = c(data$fc, data$pc), 
                  Camera = rep(c("Front Camera", "Primary Camera"), 
                               c(length(data$fc), length(data$pc))))
ggplot(cam, aes(MegaPixels, fill = Camera)) + 
  geom_bar(position = 'identity', alpha = .5)

p1 <- ggplot(data, aes(x=dual_sim, fill=dual_sim)) +
  theme_bw() +
  geom_bar() +
  ylim(0, 1250) +
  labs(title = "Dual Sim")  +
  scale_x_discrete(labels = c('Not Supported','Supported'))+
  scale_fill_manual( values = c("hotpink1", "deeppink4") )
p2 <- ggplot(data, aes(x=four_g, fill=four_g)) +
  theme_bw() +
  geom_bar() +
  ylim(0, 1250) +
  labs(title = "4 G") +
  scale_fill_manual( values = c("orchid", "orchid4") )+
  scale_x_discrete(labels = c('Not Supported','Supported'))

grid.arrange(p1, p2, nrow = 1)

p3 <- ggplot(data, aes(x=blue, fill=blue)) +
  theme_bw() +
  geom_bar() +
  ylim(0, 1250) +
  labs(title = "Bluetooth") +
  scale_fill_manual( values = c("royalblue", "midnightblue") ) +
  scale_x_discrete(labels = c('Not Supported','Supported'))
p4 <- ggplot(data, aes(x=touch_screen, fill=touch_screen)) +
  theme_bw() +
  geom_bar() +
  ylim(0, 1250) +
  labs(title = "touch screen") +
  scale_x_discrete(labels = c('Not Supported','Supported')) +
  scale_fill_manual( values = c("seagreen3", "seagreen4") )
grid.arrange(p3,p4 ,nrow=1)


### RANDOM FOREST ALGORITHM
set.seed(1234)
## defaut settings
model <- randomForest(price_range ~ ., data=data, proximity=TRUE)
model

## Number of trees
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=5),
  Type=rep(c("OOB", "très bas", "bas", "prix moyen", "prix élevé"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"très bas prix"], 
          model$err.rate[,"bas prix"],
          model$err.rate[,"prix moyen"], 
          model$err.rate[,"prix élevé"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

## repeat for 1500 trees
model <- randomForest(price_range ~ ., data=data,ntree=1500, proximity=TRUE)
model

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=5),
  Type=rep(c("OOB", "très bas", "bas", "prix moyen", "prix élevé"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"très bas prix"], 
          model$err.rate[,"bas prix"],
          model$err.rate[,"prix moyen"], 
          model$err.rate[,"prix élevé"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

### number of variables
oob.values <- vector(length=15)
for(i in 1:15) {
  temp.model <- randomForest(price_range ~ ., data=data, mtry=i, ntree=1200)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

## find the minimum error
min(oob.values)

## find the optimal value for mtry
which(oob.values == min(oob.values))

## create a model for proximities using the best value for mtry
model <- randomForest(price_range ~ ., 
                      data=data,
                      ntree=1200, 
                      proximity=TRUE, 
                      mtry=which(oob.values == min(oob.values)))
model
importance(model)
varImpPlot(model)

### K-folds cross validation

fitControl <- trainControl(method= "repeatedcv",
                           number=5, search ="random",
                           repeats=3,
                           savePrediction = T)

modelfitRF <- train(price_range ~ . , data=data, method = "rf",
                    trControl= fitControl , tuneLength= 10,
                    ntree=1000)
modelfitRF$bestTune
modelfitRF
plot(varImp(modelfitRF))

sub_rf <- subset(modelfitRF$pred, modelfitRF$pred$mtry == modelfitRF$bestTune$mtry)
caret::confusionMatrix(table(sub_rf$pred, sub_rf$obs))

#### PREDICTION
predict <- predict(modelfitRF, data = testData)
table(predict)
str(predict)
Predict <- data.frame(predict)
colnames(Predict) <- "prix"
View(Predict)
