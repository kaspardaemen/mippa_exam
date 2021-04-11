library(nycflights13)
library(tidyverse)

data <- read_delim("TrainingValidationData_200k_shuffle.csv", col_names = c('event_id', 'process_id', 'event_weight', 'MET', 'METphi', 'obj1', 'e1', 'pt1', 'eta1', 'phi1'), delim=';')

df <- data %>% arrange(event_id) %>% mutate(
  ftop = case_when(
    process_id == '4top' ~ 1,
    process_id != '4top' ~ 0
  )) %>% mutate(lMET = log(MET))
          

ggplot(data=df, aes(x=METphi)) +
  geom_histogram() +
  facet_wrap(~ftop)

mylogit <- glm(ftop ~ lMET + METphi, data = df)
summary(mylogit)
