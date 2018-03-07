########################################################################
# Data Quality Report
# 
# Author: Matthew A. Lanham
# Updated: 07/29/2014
#########################################################################

DataQualityReport = function(dataSetName) {

  n = dim(dataSetName)[[1]]       # Number of observations/records
  p = dim(dataSetName)[[2]]       # Number of attributes
  Attributes = names(dataSetName) # Attribute names
  Type = c(1:p)                   # Attributes data type
  NumberMissing = c(1:p)          # Number of missing values
  PercentComplete = c(1:p)        # Percent of missing values
  Min = c(1:p)                    # Min value for numeric attributes
  Avg = c(1:p)                    # Average value for numeric attributes
  Median = c(1:p)                 # Median value for numeric attributes
  Max = c(1:p)                    # Max value for numeric attributes
  NumberLevels = c(1:p)     # Number of Levels for factor attributes
  x = data.frame(Attributes, Type, NumberMissing, PercentComplete, Min, Avg, Median, Max, NumberLevels) # Dataframe of attribute's statistics
  
## Determine attribute type and calculate relevant statistical measures
  for (i in 1:p) {
    if (is.numeric(dataSetName[,i])) {
      x[i,2] = "numeric"
      x[i,5] = round(min(dataSetName[,i], na.rm=TRUE),2)
      x[i,6] = round(mean(dataSetName[,i], na.rm=TRUE),2)
      x[i,7] = round(median(dataSetName[,i], na.rm=TRUE),2)
      x[i,8] = round(max(dataSetName[,i], na.rm=TRUE),2)
      x[i,9] = "-"
    } else if (is.factor(dataSetName[,i])) {
      x[i,2] = "factor"
      x[i,5] = "-"
      x[i,6] = "-"
      x[i,7] = "-"
      x[i,8] = "-"
      x[i,9] = length(levels(dataSetName[,i]))
    } else if (is.integer(dataSetName[,i])) {
      x[i,2] = "integer"
      x[i,5] = "-"
      x[i,6] = "-"
      x[i,7] = "-"
      x[i,8] = "-"
      x[i,9] = "-"
    } else {
      x[i,2] = "UNKNOWN"
      x[i,5] = "-"
      x[i,6] = "-"
      x[i,7] = "-"
      x[i,8] = "-"
      x[i,9] = "-"
    }
  }
  
## Determine Number and Percentage of Missing values
for (i in 1:p) {
  x[i,3] = sum(is.na(dataSetName[i])) #Count
  x[i,4] = round(1-sum(is.na(dataSetName[i]))/n,4)*100 #Percentage
}  

# Data Quality Overall
CompleteCases = c(1:1)     # Total Complete cases
IncompleteCases = c(1:1)   # Total Incomplete cases
CompleteCasePct = c(1:1)   # Percent of Complete cases
y = data.frame(CompleteCases, IncompleteCases, CompleteCasePct) # Dataframe of attribute's statistics

y[1] = sum(complete.cases(dataSetName)) # Count of complete cases in a data frame
y[2] = sum(!complete.cases(dataSetName)) # Count of incomplete cases
y[3] = round(sum(complete.cases(dataSetName))/n,3)*100 # Count of complete cases in a data frame

# Return Data Quality Reports
return(x)
}
    
    