library(ggplot2)
library(data.table)

x <- seq(-7, 7, length.out = 100)
z <- 3.0

probs <- data.table(x = x,
                    predictability = dnorm(x, mean = z),
                    diversity = dnorm(x, sd = 2))
rews <- data.table(x = x,
                   predictability = probs[, log(predictability)],
                   diversity = probs[, -log(diversity)])
rews[, "sum" := predictability + diversity]

plotRewardTerms <- function() {
    wide <- melt(rews,  id.vars = "x")
    wide[, `Reward Term` := variable]
    ggplot(wide, aes(x = x, y = value, color = `Reward Term`)) +
        geom_line(size = 2) + xlim(-7, 7) + ylim(-10, NA) +
        xlab("g' - g") + ylab("Intrinsic Reward") +
        ggtitle("Reward Terms of G-DADS Objective for z=3") +
        geom_vline(xintercept = c(0,
                                  rews[which.max(predictability), x],
                                  rews[which.max(sum), x]),
                   linetype = "dashed", color = c("darkgreen", "red", "blue")) +
        scale_x_continuous(breaks = -7:7)
}
plotRewardTerms()

wide <- melt(probs, id.vars = "x")
wide[, Density := variable]
ggplot(wide, aes(x = x, y = value, color = Density)) +
    geom_line(size = 2) + xlab("g - g") + ylab("Probability Density") +
    ggtitle("Densities of G-DADS Objective Terms for z=3") +
    scale_x_continuous(breaks = -7:7) +
    geom_vline(xintercept = c(0, 3), linetype = "dashed",
               color = c("turquoise", "red"))
