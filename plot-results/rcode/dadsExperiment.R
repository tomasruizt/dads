library(data.table)
library(ggplot2)
# svg aspect: 500px times 350px

setwd("/tomasruiz/code/thesis/dads/")
results <- fread("plot-results/raw-data/dads-push/push-w-resampling.csv")
#results[, ITERATION := ITERATION / 500L]

name <- function(goalSpace, resampling) {
    if (!goalSpace && !resampling)
        return("DADS")
    if (resampling)
        return("Resampling Scheme")
    if (goalSpace)
        return("Goal-Space Control")
}

preprocess <- function(DT) {
    DT[, ABLATION := mapply(name, CONTROL_GOAL_SPACE, USE_RESAMPLING)]
    DT[, .(VALUE = mean(VALUE), SD = sd(VALUE)), by = .(ITERATION, ABLATION, MEASUREMENT)]
}

means <- preprocess(results)

dadsReachResults <- function() {
    color <- "blue"
    ablation <- "Goal-Space Control"
    ablation <- "Resampling Scheme"
    taskcompletion <- means[ABLATION == ablation & MEASUREMENT == "IS_SUCCESS"]
    taskcompletion <- taskcompletion[, .(ITERATION, COMPLETION=100*VALUE, SD=100*SD)]
    plotAvgTaskCompletion(DT = taskcompletion)
    print(mean(taskcompletion$COMPLETION))

    intRew <- means[ABLATION == ablation & MEASUREMENT == "PSEUDOREWARD"]
    intRew <- intRew[, .(ITERATION, PSEUDOREWARD=VALUE, SD)]
    plotIntrinsicReward(DT = intRew, color = color)
    print(mean(intRew$PSEUDOREWARD))
}
avgTaskCompletionTitle <- "Average Task Completion [%]"

cmpTaskCompletion <- function() {
    cmp <- fread("plot-results/dads-reach/full-results.csv")
    cmpMeans <- preprocess(cmp)
    taskcompletion <- rbind(means, cmpMeans)[MEASUREMENT == "IS_SUCCESS"]
    taskcompletion[, `:=`(COMPLETION=100*VALUE, SD=50*SD, ALGORITHM=ABLATION)]
    ggplot(taskcompletion, aes(x = ITERATION, y = COMPLETION, color = ALGORITHM)) +
        geom_line(size = 1) +
        geom_ribbon(aes(ymax = COMPLETION + SD, ymin = COMPLETION - SD, fill = ALGORITHM), alpha = 0.3, color = NA) +
        xlim(0, 51) + ylim(-10, 100) +
        ggtitle(avgTaskCompletionTitle)
}

plotIntrinsicReward <- function(DT, color = "blue") {
    ggplot(DT, aes(x = ITERATION, y = PSEUDOREWARD)) +
        geom_line(size = 1, color = color) +
        geom_ribbon(aes(ymax = PSEUDOREWARD + SD, ymin = PSEUDOREWARD - SD), fill = color, alpha = 0.3, color = NA) +
        ggtitle("Intrinsic Reward")
}

plotAvgTaskCompletion <- function(DT, color = "blue") {
    ggplot(DT, aes(x = ITERATION, y = COMPLETION)) +
        geom_line(size = 1, color = color) +
        geom_ribbon(aes(ymax = COMPLETION + SD, ymin = COMPLETION - SD), fill = color, alpha = 0.3, color = NA) +
        ggtitle("Average Task Completion [%]") +
        ylim(0, 100)
}

dadsPushDiagnostic <- function() {
    color <- "brown"
    measurement <- "BUFFER-TRANSITIONS_MOVING_GOAL[%]"
    measurement <- "DYNTRAIN-TRANSITIONS_MOVING_GOAL[%]"
    diagnostic <- means[ABLATION == "Resampling Scheme" & MEASUREMENT == measurement & ITERATION < 1500]
    diagnostic <- diagnostic[, .(ITERATION, PERCENTAGE=100*VALUE, SD=SD)]
    ggplot(diagnostic, aes(x = ITERATION, y = PERCENTAGE)) +
        geom_line(size = 1, color = color) +
        geom_ribbon(aes(ymax = PERCENTAGE + SD, ymin = PERCENTAGE - SD), fill = color, alpha = 0.3, color = NA) +
        ylim(0, 100) +
        ggtitle("Transitions with changing G")
}


plotPredicitonError <- function() {
    ablation <- "Resampling Scheme"
    prs <- c("PSEUDOREWARD_MOVING", "PSEUDOREWARD_NONMOVING")
    errorData <- results[MEASUREMENT %like% "DYN_L2_ERROR" | MEASUREMENT %in% prs & ABLATION == ablation]
    errorData[MEASUREMENT %like% "NONMOVING", `Transition Type` := "static"]
    errorData[!(MEASUREMENT %like% "NONMOVING"), `Transition Type` := "moving"]
    errorData[MEASUREMENT %in% prs, ]
    errorData[MEASUREMENT == "DYN_L2_ERROR_NONMOVING_GOAL", MEASUREMENT := "Non-moving transitions"]
    ggplot(errorData, aes(x = ITERATION, y = VALUE, color = MEASUREMENT)) +
        geom_point(size = 1) + ggtitle("DADS Dynamics Mean Squared Error")

}

pushComparePredictionErrorToIntrinsicReward <- function(){
    ablation <- "Resampling Scheme"
    errorData <- results[MEASUREMENT %like% "DYN_L2_ERROR" |
                             MEASUREMENT %like% "PSEUDOREWARD_" & ABLATION == ablation]
    errorData[MEASUREMENT %like% "NONMOVING", `Transition Type` := "static"]
    errorData[!(MEASUREMENT %like% "NONMOVING"), `Transition Type` := "moving"]
    errorData[MEASUREMENT %like% "DYN_L2_ERROR", METRIC := "Dynamics Prediction Error"]
    errorData[!(MEASUREMENT %like% "DYN_L2_ERROR"), METRIC := "Intrinsic Reward"]
    errorData$METRIC <- factor(errorData$METRIC, levels = c("Dynamics Prediction Error", "Intrinsic Reward"))
    errorData <- errorData[, .(ITERATION, SEED, VALUE, `Transition Type`, METRIC)]
    ggplot(errorData, aes(x = ITERATION, y = VALUE, color = `Transition Type`)) +
        facet_wrap(vars(METRIC), scales = "free") +
        ylab("") +
        geom_point(size = 1, alpha = 0.6) +
        ggtitle("High Intrinsic Reward Despite Unpredictability")

    wide <- pivot_wider(errorData[ITERATION > 100], names_from = c(METRIC), values_from = VALUE)
    ggplot(wide, aes(x = `Intrinsic Reward`, y = log(`Dynamics Prediction Error`), color = `Transition Type`)) +
        geom_point(size = 1)
}

