library(data.table)
library(ggplot2)

setwd("diayn/sac")

hand6skilld <- fread("data/hand-6skills/seed_1/progress.csv")
ggplot(hand6skilld, aes(x=episodes, y=`discrimnator-loss-mean`)) + geom_point()


skills6 <- cbind(SKILLS=6, fread("diayn-results.csv"))
skills20 <- cbind(SKILLS=20, fread("diayn-results-20-skills.csv"))
skills100 <- cbind(SKILLS=100, fread("diayn-results-100-skills.csv"))
data <- rbind(skills6, skills20, skills100)
#data <- cbind(SKILLS=6, fread("diayn-results-NoGoalHandBlock-v0-6skills.csv"))
data[, SKILLS := as.factor(SKILLS)

ggplot(data, aes(x=EPISODE, y=PSEUDOREWARD_AVG, color=SKILLS)) + geom_point()
ggplot(data, aes(x=EPISODE, y=DISCR_LOSS_AVG, color=SKILLS)) + geom_point()
