########################################################################
# Data Quality - Number of complete records in the dataset
# 
# Author: Matthew Lanham
# Updated: 07/29/2014
#########################################################################

DataQualityReportOverall = function(dataSetName) {
  
n = dim(dataSetName)[[1]]       # Number of observations/records
p = dim(dataSetName)[[2]]       # Number of attributes

# Data Quality Overall
CompleteCases = c(1:1)     # Total Complete cases
IncompleteCases = c(1:1)   # Total Incomplete cases
CompleteCasePct = c(1:1)   # Percent of Complete cases
y = data.frame(CompleteCases, IncompleteCases, CompleteCasePct) # Dataframe of attribute's statistics

y[1] = sum(complete.cases(dataSetName)) # Count of complete cases in a data frame
y[2] = sum(!complete.cases(dataSetName)) # Count of incomplete cases
y[3] = round(sum(complete.cases(dataSetName))/n,4)*100 # Count of complete cases in a data frame

# Return Data Quality Reports
return(y)
}