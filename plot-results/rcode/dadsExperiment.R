library(data.table)
library(ggplot2)
# svg aspect: 500px times 350px

setwd("/tomasruiz/code/thesis/dads/")
results <- fread("plot-results/dads-push/success-rate.csv")

name <- function(goalSpace, resampling) {
    if (!goalSpace && !resampling)
        return("DADS")
    if (goalSpace && resampling)
        return("G-DADS")
    if (goalSpace)
        return("Control Goal Space")
    return("Resampling Scheme")
}
results[["ABLATION"]] <- mapply(name, results$CONTROL_GOAL_SPACE, results$USE_RESAMPLING)
trStr <- "TRANSITIONS_MOVING_GOAL[%]"

means <- results[, .(VALUE = mean(VALUE), SD = sd(VALUE)),
                 by=.(ITERATION, ABLATION, MEASUREMENT)]

dadsReachResults <- function() {
    color <- "blue"
    taskcompletion <- means[ABLATION == "DADS" & MEASUREMENT == "IS_SUCCESS"]
    taskcompletion <- taskcompletion[, .(ITERATION, COMPLETION=100*VALUE, SD=100*SD)]
    plotAvgTaskCompletion(DT = taskcompletion)
    print(mean(taskcompletion$COMPLETION))

    intRew <- means[ABLATION == "DADS" & MEASUREMENT == "PSEUDOREWARD"]
    intRew <- intRew[, .(ITERATION, PSEUDOREWARD=VALUE, SD)]
    plotIntrinsicReward(DT = intRew, color = color)
    print(mean(intRew$PSEUDOREWARD))
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
    diagnostic <- means[ABLATION == "DADS" & MEASUREMENT == trStr & ITERATION < 1500]
    diagnostic <- diagnostic[, .(ITERATION, PERCENTAGE=100*VALUE, SD)]
    ggplot(diagnostic, aes(x = ITERATION, y = PERCENTAGE)) +
        geom_line(size = 1, color = color) +
        geom_ribbon(aes(ymax = PERCENTAGE + SD, ymin = PERCENTAGE - SD), fill = color, alpha = 0.3, color = NA) +
        ylim(0, 100) +
        ggtitle("Transitions with changing G")
}


plotData <- means[ABLATION == "DADS" & MEASUREMENT %in% c("PSEUDOREWARD", trStr)]
ggplot(plotData, aes(x = ITERATION, y = VALUE, color = ABLATION)) +
    facet_grid(MEASUREMENT ~ ABLATION, scales = "free") +
    geom_line(size = 1) +
    geom_ribbon(aes(ymax = VALUE + SD, ymin = VALUE - SD, fill = ABLATION),
                alpha = 0.3, color = NA)

transData <- means[MEASUREMENT == "PSEUDOREWARD"]
ggplot(transData, aes(x = ITERATION, y = VALUE, color = ABLATION)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymax = VALUE + SD, ymin = VALUE - SD, fill = ABLATION),
                alpha = 0.3, color = NA) + ggtitle("Transitions that move the goal [%]") +
    ylim(-1, 100)

errorData <- results[MEASUREMENT %like% "DYN_L2_ERROR" & ABLATION == "DADS"]
errorData[MEASUREMENT == "DYN_L2_ERROR_MOVING_GOAL", MEASUREMENT := "Moving transitions"]
errorData[MEASUREMENT == "DYN_L2_ERROR_NONMOVING_GOAL", MEASUREMENT := "Non-moving transitions"]
ggplot(errorData, aes(x = ITERATION, y = VALUE, color = MEASUREMENT)) +
    geom_point(size = 1) + ggtitle("DADS Dynamics Mean Squared Error")
