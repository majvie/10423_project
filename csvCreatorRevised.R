library(jsonlite)
library(here)
library(data.table)
set.seed(10423)

#set parameters
load_first_n <- 10000000
how_many_min_reviews <- 200
how_many_products <- 100
testing_prop <- 0.3
#-----------------------#

library(readr)
Reviews <- read_csv(here::here("review.csv"), progress = TRUE)
meta <- read_csv(here::here("meta.csv"), progress = TRUE)

meta = meta[,c("main_category","title","rating_number","store","parent_asin", "features", "average_rating","description","price","images")]
meta <- meta[!is.na(meta$price), ]
meta <- meta[meta$rating_number >=how_many_min_reviews, ]

#we dont want NAs
Reviews <- na.omit(Reviews)
meta <- na.omit(meta)


#the image variable seems to be nonsense and is nested, so I'm tossing it
Reviews_no_images <- Reviews[, -which(names(Reviews) == "images")]

#look at number of reviews for each product
tableReview <- table(Reviews$parent_asin)

#get to dataframe
freq_df <- as.data.frame(tableReview)
names(freq_df) <- c("parent_asin", "frequency")

#sort by frequency and take top n
top_parent_asins <- freq_df[order(-freq_df$frequency),][1:100,]

library(ggplot2)
#plot
ggplot(top_parent_asins, aes(x = reorder(parent_asin, -frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top parent_asins by Number of Reviews",
       x = "parent_asin (Product ID)",
       y = "Number of Reviews")

parent_asins_reviews_frequency <- freq_df[order(-freq_df$frequency),]
parent_asins_reviews_frequency <- parent_asins_reviews_frequency[parent_asins_reviews_frequency$frequency >= how_many_min_reviews,]

#we sort to get only those indices
Reviews_csv_at_least_min_reviews <- Reviews_no_images[which(Reviews_no_images$parent_asin %in% parent_asins_reviews_frequency$parent_asin), ]


#loop through each parent_asin
set.seed(10423)
unique_parent_asins <- unique(Reviews_csv_at_least_min_reviews$parent_asin)

unique_parent_asins_meta <- unique(meta$parent_asin)

common_parent_asins <- intersect(unique_parent_asins, unique_parent_asins_meta)

selected_products <- sample(common_parent_asins, size = how_many_products)

only_unique <- meta[meta$parent_asin %in% selected_products,]
Reviews_csv_output <- data.frame()

for (parent_asin_value in selected_products) {
  #get all reviews for parent_asin_value
  print(parent_asin_value)
  parent_asin_reviews <- Reviews_csv_at_least_min_reviews[Reviews_csv_at_least_min_reviews$parent_asin == parent_asin_value, ]
  sampled_rows <- parent_asin_reviews[sample(nrow(parent_asin_reviews), how_many_min_reviews), ]
  Reviews_csv_output <- rbind(Reviews_csv_output, sampled_rows)
}

joined_data <- merge(Reviews_csv_output, meta, by = "parent_asin", all.x = TRUE)

#lets view things to make sure its all good

#look at number of reviews for each product
tableReview2 <- table(joined_data$parent_asin)

#get to dataframe
freq_df2 <- as.data.frame(tableReview2)
names(freq_df2) <- c("parent_asin", "frequency")

#sort by frequency and take top n
top_parent_asins <- freq_df2[order(-freq_df2$frequency),]

#plot
ggplot(top_parent_asins, aes(x = reorder(parent_asin, -frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top parent_asins by Number of Reviews",
       x = "parent_asin (Product ID)",
       y = "Number of Reviews")
#if its flat, we are all good


#now for testing/training sets
testing_amount <- round(nrow(joined_data) * testing_prop)

#random sample of 30%
testing_indices <- sample(nrow(joined_data), size = testing_amount)

#all start as 1
joined_data$training <- 1
#set ones we didn't select for training as 0
joined_data$training[testing_indices] <- 0

#want mean to be 1-testing_prop
mean(joined_data$training)


fwrite(joined_data, here::here("electronics_reviews_with_meta.csv"))
















