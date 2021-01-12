library(data.table)
library(ggplot2)
library(viridisLite)

norm2 <- purrr::partial(norm, type = "2")

gdadsDT <- cbind(ALGORITHM = "GDADS-dense", fread("raw-data/ant/grid-eval-gsc.csv"))
sacDT <- cbind(ALGORITHM = "GCRL-dense (SAC)", fread("raw-data/ant/grid-eval-sac.csv"))
DT <- rbind(gdadsDT, sacDT)

DT[, NORM := round(apply(DT[, .(GOAL_X, GOAL_Y)], 1, norm2))]
plotData <- DT[, .(METRIC = mean(METRIC), SD = sd(METRIC)), by = .(NORM, ALGORITHM)]

ggplot(plotData, aes(x = NORM, y = METRIC, color = ALGORITHM)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymax = METRIC + SD, ymin = METRIC - SD, fill = ALGORITHM), alpha = 0.3, color = NA) +
    ylab("Normalized Mean Distance to Goal (∆)") +
    xlab("Initial Distance to Goal") +
    scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1.1)) +
    scale_x_continuous(breaks = seq(5, 30, by = 5)) +
    ggtitle("Performance v/s Goal Distance")

