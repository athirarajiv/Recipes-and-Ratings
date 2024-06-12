# Recipe Ratings Revealed: A Data-Driven Culinary Journey
Author: Athira Rajiv

## Introduction
In this project, I explore a comprehensive dataset of recipes and their corresponding user interactions. The main question I aim to answer is: **"What types of recipes tend to have higher average ratings?"** Gaining insight into this question can provide valuable information into culinary trends and preferences, helping home cooks who are looking to try new recipes or chefs and food bloggers aiming to create content that resonates with their audiences. The insights gained from answering this question can even reach audiences who don't usually cook, encouraging them to experiment with new ingredients and cuisines, and enriching their cooking experience. 

The dataset comprises two main files: `Recipes` and `Interactions`, containing the recipes and reviews posted since 2008. The relevant columns from these datasets include:

## Recipes

| Column          | Description                                                                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `name`          | Recipe name                                                                                                                                  |
| `id`            | Recipe ID                                                                                                                                    |
| `minutes`       | Minutes to prepare recipe                                                                                                                    |
| `contributor_id`| User ID who submitted this recipe                                                                                                            |
| `submitted`     | Date recipe was submitted                                                                                                                    |
| `tags`          | Food.com tags for the recipe                                                                                                                 |
| `nutrition`     | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for “percentage of daily value”  |
| `n_steps`       | Number of steps in the recipe                                                                                                                |
| `steps`         | Text for recipe steps, in order                                                                                                              |
| `description`   | User-provided description                                                                                                                    |

The recipes dataset contains 83,782 rows, indicating that there are 83,782 unique recipes. 

## Interactions

| Column      | Description            |
|-------------|------------------------|
| `user_id`   | User ID                |
| `recipe_id` | Recipe ID              |
| `date`      | Date of interaction    |
| `rating`    | Rating given           |
| `review`    | Review text            |
    
The ratings dataset contains 731,927 rows, indicating that there are 731,927 unique reviews for a variety of recipes.

## Data Cleaning and Exploratory Data Analysis

The data cleaning process involved several key steps to ensure the dataset was ready for analysis. First, I merged recipes and interactions on the recipe_id column to combine recipe details with user interactions. To handle missing values, I replaced ratings of 0 with np.nan as a rating of 0 likely indicates the absence of a rating rather than a legitimate score, thus preventing it from skewing the average ratings. 

Next, I performed feature engineering by creating new features from the nutrition column. Specifically, I created separate columns for each unique element in the nutrition column, so that values for 'calories', 'total_fat_PDV', 'sugar_PDV', 'sodium_PDV', 'protein_PDV', 'saturated_fat_PDV', and 'carbohydrates_PDV' are stored in different columns. In order to do this, I converted the string representations of lists in the nutrition columns into actual Python list objects. I did the same for the 'tags' column to convert the values into actual lists for easier data manipulation. Additionally, I capped the minutes column at 1440 minutes (24 hours) to remove outliers and ensure all values represented valid preparation times. I also converted the values in the submitted column to pd.datetime for consistency and better handling of date values.

Furthermore, I created an average ratings column by grouping the data by id and calculating the average rating for each recipe. This helped me incorporate a summary measure of user feedback into my analysis. Lastly, I created a has_desserts_tag column, which contains a boolean value -- the column contains True if the recipe has the "desserts" tag and False otherwise. This column is particularly relevant for the hypothesis test I conducted (see **Hypothesis Testing**) and the subsequent sections. 


My cleaned dataframe contains 83194 rows and 21 columns. I have omitted the "steps", "ingredients", and "description" columns below for better readability  of the dataset. Here are the first five rows of the cleaned DataFrame (excluding the "steps", "ingredients", and "description" columns, which is included in the actual cleaned DataFrame):

| name                                 |     id |   minutes |   contributor_id | submitted           | tags                                                                                                                                                                                                                                  | nutrition                                     |   n_steps |   n_ingredients |   average_rating |   calories |   total_fat_PDV |   sugar_PDV |   sodium_PDV |   protein_PDV |   saturated_fat_PDV |   carbohydrates_PDV | has_desserts_tag   |
|:-------------------------------------|-------:|----------:|-----------------:|:--------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|----------:|----------------:|-----------------:|-----------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|:-------------------|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27 00:00:00 | 60-minutes-or-less,time-to-make,course,main-ingredient,preparation,for-large-groups,desserts,lunch,snacks,cookies-and-brownies,chocolate,bar-cookies,brownies,number-of-servings                                                      | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]      |        10 |               9 |                4 |      138.4 |              10 |          50 |            3 |             3 |                  19 |                   6 | True               |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11 00:00:00 | 60-minutes-or-less,time-to-make,cuisine,preparation,north-american,for-large-groups,canadian,british-columbian,number-of-servings                                                                                                     | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]  |        12 |              11 |                5 |      595.1 |              46 |         211 |           22 |            13 |                  51 |                  26 | False              |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30 00:00:00 | 60-minutes-or-less,time-to-make,course,main-ingredient,preparation,side-dishes,vegetables,easy,beginner-cook,broccoli                                                                                                                 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]     |         6 |               9 |                5 |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 | False              |
| millionaire pound cake               | 286009 |       120 |           461724 | 2008-02-12 00:00:00 | time-to-make,course,cuisine,preparation,occasion,north-american,desserts,american,southern-united-states,dinner-party,holiday-event,cakes,dietary,christmas,thanksgiving,low-sodium,low-in-something,taste-mood,sweet,4-hours-or-less | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] |         7 |               7 |                5 |      878.3 |              63 |         326 |           13 |            20 |                 123 |                  39 | True               |
| 2000 meatloaf                        | 475785 |        90 |          2202916 | 2012-03-06 00:00:00 | time-to-make,course,main-ingredient,preparation,main-dish,potatoes,vegetables,4-hours-or-less,meatloaf,simply-potatoes2                                                                                                               | [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]    |        17 |              13 |                5 |      267   |              30 |          12 |           12 |            29 |                  48 |                   2 | False              |

### Univariate Analysis 

For the univariate analysis, I looked at the distribution of average recipe ratings. The plot below shows the distribution of average recipe ratings, and shows that most recipes have a high average rating, with a significant number of ratings clustering around the 4-5 range. This pattern may exist for many reasons: users may generally rate recipes positively, with fewer recipes receiving low ratings, or users are more motivated to leave positive reviews as compared to negative reviews. 


<iframe
  src="assets/average_rating_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>



### Bivariate Analysis 

The scatter plot below illustrates the relationship between preparation time (in minutes) and average rating for recipes. Darker colors indicate higher ratings, helping to visualize any potential correlation between the time invested in preparing a recipe and its perceived quality. From this plot, it seems that as the average rating decreases, the range of preparation times decreases as well. However, this is likely due to the fact that there are a significantly higher number of recipes with a 4 to 5 rating in the dataset, while there are less recipes present with ratings of 1, 2, and 3. I wanted to visualize this because I hypothesized that the longer a recipe took to make, the lower the average rating would be, as people may tend to have higher expectations for the outcome when a recipe requires more effort and time to prepare. 


<iframe
  src="assets/average_rating_vs_prep_time.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


### Interesting Aggregates 

This pivot table displays the average rating of recipes categorized by their calorie content. I defined several calorie bins to categorize the recipes based on their calorie content. The categories range from 0-100 calories to 1000+ calories. I then created a pivot table that calculates the mean average_rating for each calorie category. The pivot table is sorted in descending order of the average rating. I wanted to create this pivot table because I hoped it would help to understand if and/or how the calorie content of recipes might influence their ratings. By looking at the average ratings across different calorie categories, we can infer if there is a trend or preference among users for recipes with certain calorie levels. However, the pivot table shows that the average rating fluctuates only slightly across all the calorie categories.

<iframe
  src="assets/avg_rating_by_calories.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>

## Assessment of Missingness 

### NMAR Analysis 

### Missingness Dependency 

## Hypothesis Testing 

## Framing a Prediction Problem 

## Baseline Model 

## Final Model

## Fairness Analysis 
