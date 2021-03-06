library(nycflights13)
library(tidyverse)

data <- read_delim("TrainingValidationData_200k_shuffle.csv", col_names = c('event_id', 'process_id', 'event_weight', 'MET', 'METphi'), delim=';')

df <- data %>% arrange(event_id)

id_21 <- df %>% filter(event_id == 21)



ggplot(data=df, aes(x=MET)) +
  geom_histogram() +
  facet_wrap(~process_id)
