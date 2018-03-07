################################################################################
# Kaggle Competition - House Price Prediction
#
# PCA w/ regression and neural nets
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
################################################################################
# set memory limits
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB
memory.limit(size=10000000000024)
################################################################################
# load data
################################################################################
# note that the features (1stFlrSF,2ndFlrSF) will get automatically renamed to 
#("X1stFlrSF","X2ndFlrSF") because names in R cannot begin with numbers
setwd("C:/Purdue University/2017 Fall/XXX")
tr <- read.table("train.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2)
                             ,"numeric")
                )
te <- read.table("test.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2))
                )
################################################################################
# EDA
################################################################################
# percent of complete records using the DataQualityReportOverall() function
source("DataQualityReportOverall.R")
DataQualityReportOverall(tr)

# percent of complete records by variable using the DataQualityReport() function
source("DataQualityReport.R")
DQR = DataQualityReport(tr)
DQR_missingValue = DQR[which(DQR$NumberMissing > 0),]
DQR_missingValue

################################################################################
# Preprocess data
################################################################################
# delete the features that have more than 80% of their values missing, as well as the id column
#I should remove Id, Alley, PoolQC, Fence,MiscFeature  
drop1 <- names(tr) %in% c("Id", "Alley", "PoolQC", "Fence", "MiscFeature") 
tr <- tr[!drop1]
drop2 <- names(te) %in% c("Alley", "PoolQC", "Fence", "MiscFeature") 
te <- te[!drop2]


# for the records with missing values, impute them using the mice package using
set.seed(2016)
library(mice)
#check missing value
md.pattern(tr)
library(VIM)
aggr_plot <- aggr(tr, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(tr), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
marginplot(tr1[c(1,2)])

# mice function http://rpubs.com/skydome20/R-Note10-Missing_Value
#"m" means the number of imputation tables
#maxit means the number of iteration
tr_imp <- mice(tr,maxint=1,meth='cart',seed=2016,printFlag=F)
summary(tr_imp)
tri <- complete(tr_imp, 1)

te_imp <- mice(te,maxint=1,meth='cart',seed=2016,printFlag=F)
summary(te_imp)
tei <- complete(te_imp, 1)

save.image(file="exam2_1213.RData")
load(file="exam2_1213.RData")

################################################################################
# Zero- and Near Zero-Variance Predictors
library(caret)
dim(tri) # dimension of dataset
tri_filtered <- subset(tri, select = -c(nearZeroVar(tri, uniqueCut = 3,names=F, saveMetrics = F)))
tri <- cbind(tri$SalePrice, tri_filtered)
names(tri)[1] <- "SalePrice"
rm(tri_filtered)
# keep features in tei that were kept in tri
tei <- tei[,c("Id",names(tri)[2:(ncol(tri)-1)])]
dim(tei)
dim(tri)


################################################################################
# Creating Dummy Variables
library(caret)
# create dummies 
dummies_tri <- caret::dummyVars(SalePrice ~ ., data = tri)
dumb_tri <- data.frame(predict(dummies_tri, newdata = tri))
names(dumb_tri) <- gsub("\\.", "", names(dumb_tri))
tri <- dumb_tri
rm(dummies_tri,dumb_tri)

tri$SalePrice = cbind(tr$SalePrice,tri)


# create dummies on score set
dummies_tei <- dummyVars(~ ., data = tei)
ex <- data.frame(predict(dummies_tei, newdata = tei))
names(ex) <- gsub("\\.", "", names(ex)) # removes dots from col names
tei <- ex
rm(dummies_tei, ex)

# ensure only features available in both train and score sets are kept
tri <- tri[ , c("SalePrice", Reduce(intersect, list(names(tri), names(tei))))]
tei <- tei[ , c("Id", Reduce(intersect, list(names(tri), names(tei))))]

dim(tri)
dim(tei)

################################################################################
# Identify Correlated Predictors and remove them

# correlation matrix
descrCor <-  cor(tri[,2:ncol(tri)]) 
# number of features having a correlation greater than some value
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.90)
# remove those specific columns from your dataset
filteredDescr <- tri[,2:ncol(tri)][,-highlyCorDescr] 
# create a new correlation matrix
descrCor2 <- cor(filteredDescr)
#check those correlations of all features within 0.9
summary(descrCor[upper.tri(descrCor2)])

#put predicting variable back
tri <- cbind(tri$SalePrice, filteredDescr)
names(tri)[1] <- "SalePrice"
rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)
# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

dim(tei)
dim(tri)

################################################################################
# Identifying Linear Dependencies and remove them
# Find if any linear combinations exist and which column combos they are
comboInfo <- caret::findLinearCombos(tri)
tri <- tri[,-comboInfo[2]$remove] 
rm(comboInfo)
# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]
tri <- tri[,c("SalePrice", Reduce(intersect, list(names(tri), names(tei))))]
dim(tri)
dim(tei)


################################################################################
# The remaining features that are truely categorical features, make sure they are
# defined as factors. Below are all the possibilities that you might get using
# dummyVars()
str(tri)
names(tri)
cols <- c("Id","MSSubClass120","MSSubClass160","MSSubClass180","MSSubClass190",
"MSSubClass20","MSSubClass30","MSSubClass40","MSSubClass45","MSSubClass50",
"MSSubClass60","MSSubClass70","MSSubClass75","MSSubClass80","MSSubClass85",
"MSSubClass90","MSZoningCall","MSZoningFV","MSZoningRH","MSZoningRL","MSZoningRM",
"StreetGrvl","StreetPave","AlleyGrvl","AlleyPave","LotShapeIR1","LotShapeIR2",
"LotShapeIR3","LotShapeReg","LandContourBnk","LandContourHLS","LandContourLow",
"LandContourLvl","UtilitiesAllPub","UtilitiesNoSeWa","LotConfigCorner","LotConfigCulDSac",
"LotConfigFR2","LotConfigFR3","LotConfigInside","LandSlopeGtl","LandSlopeMod",
"LandSlopeSev","NeighborhoodBlmngtn","NeighborhoodBlueste","NeighborhoodBrDale",
"NeighborhoodBrkSide","NeighborhoodClearCr","NeighborhoodCollgCr","NeighborhoodCrawfor",
"NeighborhoodEdwards","NeighborhoodGilbert","NeighborhoodIDOTRR","NeighborhoodMeadowV",
"NeighborhoodMitchel","NeighborhoodNAmes","NeighborhoodNoRidge","NeighborhoodNPkVill",
"NeighborhoodNridgHt","NeighborhoodNWAmes","NeighborhoodOldTown","NeighborhoodSawyer",
"NeighborhoodSawyerW","NeighborhoodSomerst","NeighborhoodStoneBr","NeighborhoodSWISU",
"NeighborhoodTimber","NeighborhoodVeenker","Condition1Artery","Condition1Feedr",
"Condition1Norm","Condition1PosA","Condition1PosN","Condition1RRAe","Condition1RRAn",
"Condition1RRNe","Condition1RRNn","Condition2Artery","Condition2Feedr","Condition2Norm",
"Condition2PosA","Condition2PosN","Condition2RRAe","Condition2RRAn","Condition2RRNn",
"BldgType1Fam","BldgType2fmCon","BldgTypeDuplex","BldgTypeTwnhs","BldgTypeTwnhsE",
"HouseStyle15Fin","HouseStyle15Unf","HouseStyle1Story","HouseStyle25Fin","HouseStyle25Unf",
"HouseStyle2Story","HouseStyleSFoyer","HouseStyleSLvl","RoofStyleFlat","RoofStyleGable",
"RoofStyleGambrel","RoofStyleHip","RoofStyleMansard","RoofStyleShed","RoofMatlClyTile",
"RoofMatlCompShg","RoofMatlMembran","RoofMatlMetal","RoofMatlRoll","RoofMatlTarGrv",
"RoofMatlWdShake","RoofMatlWdShngl","Exterior1stAsbShng","Exterior1stAsphShn",
"Exterior1stBrkComm","Exterior1stBrkFace","Exterior1stCBlock","Exterior1stCemntBd",
"Exterior1stHdBoard","Exterior1stImStucc","Exterior1stMetalSd","Exterior1stPlywood",
"Exterior1stStone","Exterior1stStucco","Exterior1stVinylSd","Exterior1stWdSdng",
"Exterior1stWdShing","Exterior2ndAsbShng","Exterior2ndAsphShn","Exterior2ndBrkCmn",
"Exterior2ndBrkFace","Exterior2ndCBlock","Exterior2ndCmentBd","Exterior2ndHdBoard",
"Exterior2ndImStucc","Exterior2ndMetalSd","Exterior2ndOther","Exterior2ndPlywood",
"Exterior2ndStone","Exterior2ndStucco","Exterior2ndVinylSd","Exterior2ndWdSdng",
"Exterior2ndWdShng","MasVnrTypeBrkCmn","MasVnrTypeBrkFace","MasVnrTypeNone",
"MasVnrTypeStone","ExterQualEx","ExterQualFa","ExterQualGd","ExterQualTA",
"ExterCondEx","ExterCondFa","ExterCondGd","ExterCondPo","ExterCondTA","FoundationBrkTil",
"FoundationCBlock","FoundationPConc","FoundationSlab","FoundationStone","FoundationWood",
"BsmtQualEx","BsmtQualFa","BsmtQualGd","BsmtQualTA","BsmtCondFa","BsmtCondGd",
"BsmtCondPo","BsmtCondTA","BsmtExposureAv","BsmtExposureGd","BsmtExposureMn",
"BsmtExposureNo","BsmtFinType1ALQ","BsmtFinType1BLQ","BsmtFinType1GLQ","BsmtFinType1LwQ",
"BsmtFinType1Rec","BsmtFinType1Unf","BsmtFinType2ALQ","BsmtFinType2BLQ","BsmtFinType2GLQ",
"BsmtFinType2LwQ","BsmtFinType2Rec","BsmtFinType2Unf","HeatingGasA","HeatingGasW",
"HeatingGrav","HeatingOthW","HeatingWall","HeatingQCEx","HeatingQCFa","HeatingQCGd",
"HeatingQCPo","HeatingQCTA","CentralAirN","CentralAirY","ElectricalFuseA",
"ElectricalFuseF","ElectricalFuseP","ElectricalMix","ElectricalSBrkr","KitchenQualEx",
"KitchenQualFa","KitchenQualGd","KitchenQualTA","FunctionalMaj1","FunctionalMaj2",
"FunctionalMin1","FunctionalMin2","FunctionalMod","FunctionalSev","FunctionalTyp",
"FireplaceQuEx","FireplaceQuFa","FireplaceQuGd","FireplaceQuPo","FireplaceQuTA",
"GarageType2Types","GarageTypeAttchd","GarageTypeBasment","GarageTypeBuiltIn",
"GarageTypeCarPort","GarageTypeDetchd","GarageFinishFin","GarageFinishRFn",
"GarageFinishUnf","GarageQualEx","GarageQualFa","GarageQualGd","GarageQualPo",
"GarageQualTA","GarageCondEx","GarageCondFa","GarageCondGd","GarageCondPo",
"GarageCondTA","PavedDriveN","PavedDriveP","PavedDriveY","PoolQCEx","PoolQCFa",
"PoolQCGd","FenceGdPrv","FenceGdWo","FenceMnPrv","FenceMnWw","MiscFeatureGar2",
"MiscFeatureOthr","MiscFeatureShed","MiscFeatureTenC","SaleTypeCOD","SaleTypeCon",
"SaleTypeConLD","SaleTypeConLI","SaleTypeConLw","SaleTypeCWD","SaleTypeNew",
"SaleTypeOth","SaleTypeWD","SaleConditionAbnorml","SaleConditionAdjLand",
"SaleConditionAlloca","SaleConditionFamily","SaleConditionNormal","SaleConditionPartial")
cols <- Reduce(intersect, list(names(tri), cols))
tri[cols] <- lapply(tri[cols], factor)
tei[cols] <- lapply(tei[cols], factor)

save.image(file="exam2_QQQ9.RData")
load(file="exam2_QQQ9.RData")

################################################################################
# standardize the input features using the preProcess() using a min-max normalization 
# (aka "range"), in addition to using a"YeoJohnson" transformation to make the 
# features more bell-shaped
# call this pre-processed data set "trit"
set.seed(1234)

#remove SalePrice and ID
tri_noSalePrice <- tri[, 2:(ncol(tri))]
tei_noID <- tei[, 2:(ncol(tei))]

#set preprocess vales by min-max normalization and Yeo Johnson
preproessValue <- preProcess(tri_noSalePrice, method = c("range", "YeoJohnson"))

# transform the data sets
trit <- predict(preproessValue, tri_noSalePrice)
trit = cbind(tri$SalePrice,trit)
names(trit)[1]<-paste("SalePrice")

teit <- predict(preproessValue, tei_noID)
teit = cbind(tei$Id,teit)
names(teit)[1]<-paste("Id")


################################################################################
# These are the features I have so far
dim(trit) #190
dim(teit) #190
#
# subset your trit dataset by creating a new dataset called trit_num, which
# only contains the numeric features (INCLUDING your target variable)
nums <- sapply(trit, is.numeric)
trit_num <- trit[ ,nums]
# subset your teit dataset by creating a new dataset called teit_num, which
# only contains the numeric features (INCLUDING your target variable)
nums2 <- sapply(teit, is.numeric)
teit_num <- teit[ ,nums2]


save.image(file="exam2_Q12.RData")
load(file="exam2_Q12.RData")

################################################################################
# Dimension Reduction - PCA
################################################################################
# createDataOartition
set.seed(1234)
library(caret)
# select randomly 50% of the dataset to serve as training data
inTrain<- createDataPartition(y = trit_num$SalePrice, times =1, p=0.5, list=FALSE)  
trit_num_train <- trit_num[inTrain,]
trit_num_test <- trit_num[-inTrain,]

#remove SalePrice
trit_num_train <- trit_num_train[, 2:(ncol(trit_num_train))]
trit_num_test <- trit_num_test[, 2:(ncol(trit_num_test))]

# perform PCA using the principal() function from the psych package on the train set
library(psych)
pca1 <- principal(trit_num_train
                  , nfactors = 15     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T )      # find component scores or not

# obtain eignvalues
pca1$values

# How much does the first component account for of the total variance in this dataset?
pca1$loadings
#23.4%

# Based on the proportion of variance explained criterion, how many components would
# you keep based on a 65% cutoff?
# I should keep 8 components


# Using the eigenvalue criterion how many components would you keep?
#eigenvale > 1 should be retained
#I should keep 8 componets



# generate a scree plot and decide how many components to keep?
par(mfrow=c(1,1))
plot(pca1$values, type="b", main="Scree plot for Q18", col="blue")


# perform validation of pca on your test set using the # of PCs you believe makes
# sense based on what the previous 3 criterion led to you choose. Based on the these
# results ONLY (without profiling) do you believe this PCA is repeatable?
pca2 <- principal(trit_num_test
                  , nfactors = 8     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)
pca2$loadings



################################################################################
# Create PCs as input features to be used for prediction
################################################################################
# Now, run PCA on the entire "trit_num" dataset (which ignores the target), using
# the number of components that are features in the dataset
pca_tr <- principal(trit_num[,2:ncol(trit_num)]
                  , nfactors = 8    # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)

# repeat this step to obtain the pc rotations for the teit_num dataset
pca_te <- principal(teit_num[,2:ncol(teit_num)]
                 , nfactors = 8     # number of componets to extract
                 , rotate = "none"  # can specify different rotations
                 , scores = T       # find component scores or not
)

# obtain the prinipcal component scores so we can use those as possible
# input features later. Call the PC scores for the training data "tr_pcscores"
# and the PC scores on the scoring set "te_pcscores"
tr_pcscores <- data.frame(predict(pca_tr, data=trit_num[,2:ncol(trit_num)]))
te_pcscores <- data.frame(predict(pca_te, data=teit_num[,2:ncol(teit_num)]))
# just keep the first x # of columns (PCs) based on your decision in how many
# to retain from previous analyses. Hint: You should not be keeping 29 columns.
tr_pcscores <- tr_pcscores[,1:8]
te_pcscores <- te_pcscores[,1:8]

################################################################################
# Make sure datasets for all experiments are standardized similarly
################################################################################
# We are going to use three different datasets and try various approaches
# to modeling to see what works best for this regression-type problem
# 1) trit: imputed data with min-max and Yeo-Johnson transformations
#    teit is our scoring set        
# 2) tr_pcscores: retained principal components as features
#    te_pcscores is our scoring set

# On the tr_pcscores and te_pcscores datasets, perform a mix-max normalization 
# (aka range) and YeoJohnson transformation and overwrite (i.e. call them the same 
# names) those datasets with those standardized values.
preProcValues <- preProcess(tr_pcscores[,1:ncol(tr_pcscores)], method = c("range","YeoJohnson"))
tr_pcscores <- predict(preProcValues, tr_pcscores)
preProcValues <- preProcess(te_pcscores[,1:ncol(te_pcscores)], method = c("range","YeoJohnson"))
te_pcscores <- predict(preProcValues, te_pcscores)

# 3) tr_scoresNfactors: retained principal components + factor features
#    tr_scoresNfactors is our scoring set
facs <- sapply(trit, is.factor)
trit_facs <- trit[ ,facs]
tr_scoresNfactors <- data.frame(tr_pcscores, trit_facs)
facs2 <- sapply(teit, is.factor)
teit_facs <- teit[ ,facs2]
te_scoresNfactors <- data.frame(te_pcscores, teit_facs)

# ensure the target variables is in the new datasets
tr_pcscores <- data.frame(trit$SalePrice, tr_pcscores); names(tr_pcscores)[1] <- "SalePrice"
tr_scoresNfactors <- data.frame(trit$SalePrice, tr_scoresNfactors); 
names(tr_scoresNfactors)[1] <- "SalePrice"

################################################################################
# R Environment cleanup 
################################################################################
# At this point you probably have alot of variables in your R environment. Lets
# clean this up these items we don't need anymore
rm(imputedValues, imputedValues2, pca1, pca2, preProcValues,trainIndex, trit_facs
   , trit_num, facs, facs2, DataQualityReport, DataQualityReportOverall, pca_te
   , pca_tr, teit_facs, teit_num, train, test, pca, nums, nums2)

################################################################################
# Model building 
################################################################################
# Make sure you set a seed of 1234 before partitioning your data into train
# and test sets.
set.seed(1234)
library(caret)

# create an 80/20 train and test set for each of the three "tr" datasets (trit
#, tr_pcscores, tr_scoresNfactors) using the createDataPartition() function
trainIndex <- createDataPartition(trit$SalePrice # target variable vector
                                  , p = 0.80    # % of data for training
                                  , times = 1   # Num of partitions to create
                                  , list = F )   # should result be a list (T/F)

#train1
set.seed(1234)
train1 <- trit[trainIndex,]
test1 <- trit[-trainIndex,]
#train2
set.seed(1234)
train2 <- tr_pcscores[trainIndex,]
test2 <- tr_pcscores[-trainIndex,]
#train3
set.seed(1234)
train3 <- tr_scoresNfactors[trainIndex,]
test3 <- tr_scoresNfactors[-trainIndex,]


################################################################################
# multiple linear regression (forward and backward selection) on trit (train1) dataset
################################################################################
#Use the regsubsets() function from the leaps package to perform forward selection and 
#backward selection on train1. Based on the features kept in each case, run individual 
#models using lm() to train a model. Call the model using the features from forward 
#selection m1f and the model using the features from backward selection m1b.
m1f <- lm(SalePrice ~ ., data=train1)
summary(m1f)

# forward selection
library(leaps)
mlf <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='forward') #plot(mlf)
vars2keep <- data.frame(summary(mlf)$which[which.max(summary(mlf)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep
#[1] "LotArea"              "NeighborhoodNoRidge1" "OverallQual"          "YearRemodAdd"        
#[5] "BsmtQualEx1"          "BsmtFinSF1"           "GrLivArea"            "KitchenQualEx1"      
#[9] "GarageCars"   
modelFormula <- paste("SalePrice ~ LotArea+NeighborhoodNoRidge+OverallQual+YearRemodAdd+BsmtQualEx+
                      BsmtFinSF1+GrLivArea+KitchenQualEx+GarageCars")
m1f <- lm(modelFormula, data=train1)
summary(m1f)

# backward selection
library(leaps)
mlb <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='backward') #plot(mlf)
vars2keep <- data.frame(summary(mlb)$which[which.max(summary(mlb)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep
#[1] "LotArea"              "NeighborhoodNoRidge1" "OverallQual"          "BsmtQualEx1"         
#[5] "BsmtExposureGd1"      "BsmtFinSF1"           "GrLivArea"            "KitchenQualEx1"      
#[9] "GarageCars"  
modelFormula_b <- paste("SalePrice ~ LotArea+NeighborhoodNoRidge+OverallQual+BsmtQualEx+
                        BsmtExposureGd+BsmtFinSF1+GrLivArea+KitchenQualEx+GarageCars")
m1b <- lm(modelFormula_b, data=train1)
summary(m1b)

# perform regression diagnostics for forward and backward selection models and 
# discuss if you see any potential issues or assumption violations.
source("myDiag.R")
myDiag(m1f)
myDiag(m1b)

# plot predicted vs actual
par(mfrow=c(1,2))
yhat_m1f <- predict(m1f, newdata=train1); plot(train1$SalePrice, yhat_m1f)
yhat_m1b <- predict(m1b, newdata=train1); plot(train1$SalePrice, yhat_m1b)

################################################################################
# multiple linear regression on tr_pcscores (train2) dataset
################################################################################
# Use lm() to create a multiple linear regression model using on the PCs as inputs. 
# In other words, use the train2 dataset. Call this model m2. Provide regression 
# diagnostics plots and discuss if you see any potential issues or assumptions violated.
m2 <- lm(SalePrice ~ ., data=train2)
summary(m2)

# plot predicted vs actual
par(mfrow=c(1,1))
yhat_m2 <- predict(m2, newdata=train2); plot(train2$SalePrice, yhat_m2)

################################################################################
# multiple linear regression on tr_scoresNfactors (train3) dataset
################################################################################
# Use lm() function to create a multiple linear regression model on all features 
# from the train3 dataset. Call this model m3.
m3 <- lm(SalePrice ~ ., data=train3)
summary(m3)


# backward selection
library(leaps)
m2b <- regsubsets(SalePrice ~ ., data=train3, nbest=1, intercept=T, method='backward')
vars2keep <- data.frame(summary(m2b)$which[which.max(summary(m2b)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep
#[1] "PC1"                  "PC3"                  "NeighborhoodNoRidge1" "NeighborhoodStoneBr1"
#[5] "ExterQualEx1"         "ExterQualGd1"         "BsmtQualEx1"          "BsmtExposureGd1"     
#[9] "KitchenQualEx1" 

modelFormula_b3 <- paste("SalePrice ~ PC1+PC3+NeighborhoodNoRidge+NeighborhoodStoneBr+
                         ExterQualEx+ExterQualGd+BsmtQualEx+BsmtExposureGd+KitchenQualEx")
m3b <- lm(modelFormula_b3, data=train3)
summary(m3b)


# perform regression diagnostics and discuss if you see any potential issues or 
# assumption violations.
source("myDiag.R")
myDiag(m3b)

# plot predicted vs actual
par(mfrow=c(1,2))
yhat_m3 <- predict(m3, newdata=train3); plot(train3$SalePrice, yhat_m3)
yhat_m3b <- predict(m3b, newdata=train3); plot(train3$SalePrice, yhat_m3b)

################################################################################
# Neural Networks
################################################################################
# Use trainControl() to set up a 3-fold cross-validation modeling design for a regression-type problem.
library(caret)
set.seed(1234)
ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number= 3      # k number of times to do k-fold
                     
)


(maxvalue <- summary(trit$SalePrice)["Max."][[1]])

nnet1 <- train(SalePrice/755000 ~ LotArea+NeighborhoodNoRidge+OverallQual+YearRemodAdd+
                 BsmtQualEx+BsmtFinSF1+BsmtExposureGd+GrLivArea+KitchenQualEx+GarageCars,       
                  data = train1,     # training set used to build model
                  method = "nnet",     # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  tuneLength = 15,
                  maxit = 100,
                  metric = "RMSE"     # performance measure
)



# This code shows your "best" tuning parameters for this neural network.
nnet1$finalModel$tuneValue
#size       decay
#37    5 0.001425103

# Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.
myGrid <-  expand.grid(size = c(3,4,5,6,7)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.001425103
                                   ,0.011425103))  #parameter for weight decay. 
nnet1b <- train(SalePrice/755000 ~ LotArea+NeighborhoodNoRidge+OverallQual+YearRemodAdd
                +BsmtQualEx+BsmtFinSF1+GrLivArea+KitchenQualEx+GarageCars,
               data = train1,     # training set used to build model 
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneGrid = myGrid,
               maxit = 500,
               metric = "RMSE"     # performance measure
)

par(mfrow=c(1,2))
yhat_nn1 <- predict(nnet1, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1)
yhat_nn1b <- predict(nnet1b, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1b)


nnet1b$finalModel$tuneValue


# Neural network on train2
library(caret)
set.seed(1234)
nnet2 <- train(SalePrice/755000 ~ .,       # model specification
               data = train2,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE")     # performance measure


nnet2$finalModel$tuneValue
#size        decay
#80   11 0.0004923883


# Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.
myGrid <-  expand.grid(size = c(9,10,11,12,13)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.0004923883
                                   ,0.0104923883))  #parameter for weight decay. 
nnet2b <- train(SalePrice/755000 ~ .,
                data = train2,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid,
                maxit = 500,
                metric = "RMSE"     # performance measure
) 

nnet2b$finalModel$tuneValue
#size      decay
#15   13 0.01049239

par(mfrow=c(1,2))
yhat_nn2 <- predict(nnet2, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2)
yhat_nn2b <- predict(nnet2b, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2b)

# Neural network on train3
# LINES OF NEEDED CODE MISSING HERE - STUDENT NEEDS TO FIGURE OUT WHAT GOES HERE
library(caret)
set.seed(1234)
nnet3 <- train(SalePrice/755000 ~ PC1+PC3+NeighborhoodNoRidge+NeighborhoodStoneBr+
                 ExterQualEx+ExterQualGd+BsmtQualEx+BsmtExposureGd+KitchenQualEx ,     
               data = train3,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE" )    # performance measure



nnet3$finalModel$tuneValue
#size        decay
#33    5 0.0001701254

# Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.
myGrid <-  expand.grid(size = c(3,4,5,6,7)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.0001701254
                                   ,0.0101701254))  #parameter for weight decay. 
nnet3b <- train(SalePrice/755000 ~ PC1+PC3+NeighborhoodNoRidge+NeighborhoodStoneBr+
                  ExterQualEx+ExterQualGd+BsmtQualEx+BsmtExposureGd+KitchenQualEx,
                data = train3,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid,
                maxit = 500,
                metric = "RMSE"     # performance measure
)

nnet3b$finalModel$tuneValue
#size      decay
#3    3 0.01017013
yhat_nn3 <- predict(nnet3, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3)
yhat_nn3b <- predict(nnet3b, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3b)


################################################################################
# Decision Trees
################################################################################
# 
library(tree)
tree1 = tree(SalePrice ~ .
               , control = tree.control(nobs=nrow(train1)[[1]]
                                        , mincut = 0
                                        , minsize = 1
                                        , mindev = 0.01)
               , data = train1)
summary(tree1)

par(mfrow=c(1,1))
plot(tree1); text(tree1, pretty=0) # plot the tree

# perform cross-validation to find optimal number of terminal nodes
cv.tree1 = cv.tree(tree1)
par(mfrow=c(1,1))
plot(cv.tree1$size
     , cv.tree1$dev
     , type = 'b')

# prune tree where the number of terminal nodes is ...
prunedfit = prune.tree(tree1, best=3)
summary(prunedfit)
plot(prunedfit); text(prunedfit, pretty=0)



yhat_tree1 <- predict(tree1, newdata=train1); plot(train1$SalePrice, yhat_tree1)
yhat_tp <- predict(prunedfit, newdata=train1); plot(train1$SalePrice, yhat_tp)



## bagged tree on train1
tree1b <- train(SalePrice ~ .,
               data = train1,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

## bagged tree on train2
tree2 <- train(SalePrice ~ .,
               data = train2,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

## bagged tree on train3
tree3 <- train(SalePrice ~ .,
               data = train3,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

par(mfrow=c(2,2))
yhat_dt1 <- predict(tree1, newdata=train1); plot(train1$SalePrice, yhat_dt1)
yhat_dt1b <- predict(tree1b, newdata=train1); plot(train1$SalePrice, yhat_dt1b)
yhat_dt2 <- predict(tree2, newdata=train2); plot(train2$SalePrice, yhat_dt2)
yhat_dt3 <- predict(tree3, newdata=train3); plot(train3$SalePrice, yhat_dt3)

################################################################################
# Model Evaluation
################################################################################

yhat_m1f_te <- predict(m1f, newdata=test1)
yhat_m1b_te <- predict(m1b, newdata=test1)
yhat_m2_te <- predict(m2, newdata=test2)
yhat_m3_te <- predict(m3, newdata=test3)
yhat_m3b_te <- predict(m3b, newdata=test3)

yhat_nn1_te <- predict(nnet1, newdata=test1)*maxvalue
yhat_nn1b_te <- predict(nnet1b, newdata=test1)*maxvalue
yhat_nn2_te <- predict(nnet2, newdata=test2)*maxvalue
yhat_nn2b_te <- predict(nnet2b, newdata=test2)*maxvalue
yhat_nn3_te <- predict(nnet3, newdata=test3)*maxvalue
yhat_nn3b_te <- predict(nnet3b, newdata=test3)*maxvalue

yhat_dt1_te <- predict(tree1, newdata=test1);
yhat_dt1b_te <- predict(tree1b, newdata=test1)
yhat_dt2_te <- predict(tree2, newdata=test2)
yhat_dt3_te <- predict(tree3, newdata=test3)


results <- matrix(rbind(
cbind(t(postResample(pred=yhat_m1f, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_m1f_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m1b, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_m1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m2, obs=train2$SalePrice)), 
      t(postResample(pred=yhat_m2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_m3, obs=train3$SalePrice)), 
      t(postResample(pred=yhat_m3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_m3b, obs=train3$SalePrice)), 
      t(postResample(pred=yhat_m3b_te, obs=test3$SalePrice))),
  
cbind(t(postResample(pred=yhat_nn1, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_nn1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn1b, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_nn1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn2, obs=train2$SalePrice)), 
      t(postResample(pred=yhat_nn2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn2b, obs=train2$SalePrice)), 
      t(postResample(pred=yhat_nn2b_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn3, obs=train3$SalePrice)), 
      t(postResample(pred=yhat_nn3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_nn3b, obs=train3$SalePrice)), 
      t(postResample(pred=yhat_nn3b_te, obs=test3$SalePrice))),
    
cbind(t(postResample(pred=yhat_dt1, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_dt1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt1b, obs=train1$SalePrice)), 
      t(postResample(pred=yhat_dt1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt2, obs=train2$SalePrice)), 
      t(postResample(pred=yhat_dt2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_dt3, obs=train3$SalePrice)), 
      t(postResample(pred=yhat_dt3_te, obs=test3$SalePrice)))
), nrow=15)
colnames(results) <- c("Train_RMSE", "Train_R2","Test_RMSE", "Test_R2")
rownames(results) <- c("MLR_Forward","MLR_Backward","MLR_PCs","MLR_PCs+Factors",
                       "MLR_Backward_PCs+Factors","NN_ForBackFeatures","NN_ForBackFeatures_Optimized",
                       "NN_PCs","NN_PCs_Optimized","NN_BackFeatures","NN_BackFeatures_Optimized",
                       "Tree_Numerics+Factors","BaggedTree_Numerics+Factors",
                       "BaggedTree_PCs","BaggedTree_PCs+Factors")
results

library(reshape2)
results <- melt(results)
names(results) <- c("Model","Stat","Values")


library(ggplot2)
# RMSE
p1 <- ggplot(data=results[which(results$Stat=="Train_RMSE" | results$Stat=="Test_RMSE"),]
            , aes(x=Model, y=Values, fill=Stat)) 
p1 <- p1 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p1 <- p1 + facet_grid(~Model, scale='free_x', drop = TRUE)
p1 <- p1 + scale_fill_manual(values=c('#FF6666','blue'))
p1 <- p1 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p1 <- p1 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
               strip.text.y = element_text(size=0, face="bold"),
               strip.background = element_rect(colour="white", fill="white"))
p1 <- p1 + ggtitle("RMSE Performance")
p1

# R2
p2 <- ggplot(data=results[which(results$Stat=="Train_R2" | results$Stat=="Test_R2"),]
             , aes(x=Model, y=Values, fill=Stat)) 
p2 <- p2 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p2 <- p2 + facet_grid(~Model, scale='free_x', drop = TRUE)
p2 <- p2 + scale_fill_manual(values=c('#FF6666','blue'))
p2 <- p2 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p2 <- p2 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
                 strip.text.y = element_text(size=0, face="bold"),
                 strip.background = element_rect(colour="white", fill="white"))
p2 <- p2 + ggtitle("R2 Performance")
p2

################################################################################
# Score data / Deployment
################################################################################

#This is the best model
yhat_nn1b_teit <- predict(nnet1b, newdata=teit)*maxvalue
submission <- data.frame(teit$Id,yhat_nn1b_teit)
colnames(submission) <- c("Id","SalePrice")

# Write out file to be uploaded to Kaggle.com for scoring
write.table(submission, file = "SalePrice_prediction_ShanLin.csv", 
            quote=F, sep=",", row.names=F, col.names=T)


