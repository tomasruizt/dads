library(data.table)
library(ggplot2)
library(viridisLite)

norm2 <- purrr::partial(norm, type = "2")

gdadsDT <- cbind(ALGORITHM = "G-DADS-dense", fread("raw-data/ant/grideval2/gdads.csv"))
sacDT <- cbind(ALGORITHM = "GCRL-dense (SAC)", fread("raw-data/ant/grideval2/sac.csv"))
dadsDT <- data.table(ALGORITHM = "DADS-dense", NORM = 6:30,
                     METRIC = c(0.75, 0.4, 0.45, 0.38, 0.35,
                                0.42, 0.4, 0.42, 0.41, 0.4,
                                0.48, 0.5, 0.45, 0.48, 0.65,
                                0.5, 0.4, 0.55, 0.5, 0.55,
                                0.52, 0.55, 0.58, 0.65, 0.55),
                     SD = c(0.20, 0.1, 0.15, 0.05, 0.05,
                            0.1, 0.15, 0.1, 0.075, 0.05,
                            0.1, 0.15, 0.1, 0.1, 0.2,
                            0.1, 0.05, 0.08, 0.1, 0.1,
                            0.1, 0.08, 0.05, 0.1, 0.06))

DT <- rbind(gdadsDT, sacDT)

DT[, NORM := round(apply(DT[, .(GOAL_X, GOAL_Y)], 1, norm2))]
plotData <- DT[, .(METRIC = mean(METRIC), SD = sd(METRIC)), by = .(NORM, ALGORITHM)]
plotData <- rbind(plotData, dadsDT)
colorProgression <- c("blue4", "green3", "red3")

ggplot(plotData[NORM > 5], aes(x = NORM, y = METRIC, color = ALGORITHM)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymax = METRIC + SD, ymin = METRIC - SD, fill = ALGORITHM), alpha = 0.2, color = NA) +
    ylab("Normalized Mean Distance to Goal (∆)") +
    xlab("Initial Distance to Goal") +
    scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1.5)) +
    scale_x_continuous(breaks = seq(5, 30, by = 5)) +
    scale_color_manual(values = colorProgression) +
    scale_fill_manual(values = colorProgression) +
    ggtitle("Performance v/s Goal Distance")

