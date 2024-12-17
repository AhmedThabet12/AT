-- The project aims to highlight the role of difference factors on the overall ride counts

-- Task 1: The most frequent service users' demographics

WITH t1 AS(

SELECT
r.customer_id,
COUNT(r.ride_id) AS rides_count,
cu.age,
CASE
WHEN cu.age BETWEEN 18 and 29 THEN '18-29'
WHEN cu.age BETWEEN 30 and 44 THEN '30-44'
WHEN cu.age BETWEEN 45 and 59 THEN '45-59'
ELSE '60+'
END AS age_group,
cu.gender,
NTILE(5) OVER (ORDER BY COUNT(r.ride_id)) AS rides_quintiles -- grouping the rides into 5 groups
FROM rides r
LEFT JOIN customers cu
ON cu.customer_id = r.customer_id
GROUP BY
r.customer_id,
cu.age,
cu.gender
ORDER BY rides_count DESC
)
SELECT
age_group,
gender,
COUNT(rides_count) AS rides_count,
COUNT(customer_id) AS customer_count
FROM t1
WHERE rides_quintiles IN (5)

GROUP BY age_group,gender
ORDER BY age_group,gender;

-- Task 2: Membership Status (Premium and Regular)

SELECT
c.membership_status,
COUNT(r.ride_id) AS rides_count,
ROUND(AVG(r.duration),3) AS average_duration,
PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.duration) AS median_duration,
ROUND(AVG(r.fare),3) AS average_fare,
ROUND(sum(r.fare) / NULLIF(sum(r.duration),0),3) AS fare_per_min,
ROUND(sum(r.fare) / NULLIF(sum(r.distance),0),3) AS fare_per_km
FROM rides r
JOIN customers c
ON r.customer_id = c.customer_id
GROUP BY c.membership_status

-- Task 3: Average ride duration, Average fare and rides count by age group

WITH t1 AS(
SELECT
c.age,
CASE
WHEN c.age BETWEEN 18 and 29 THEN '18-29'
WHEN c.age BETWEEN 30 and 44 THEN '30-44'
WHEN c.age BETWEEN 45 and 59 THEN '45-59'
ELSE '60+'
END AS age_group,
r.duration,
r.fare,
r.distance,
r.ride_id
FROM rides r
JOIN customers c
ON r.customer_id = c.customer_id
)
SELECT age_group,
ROUND(AVG(duration),3) AS average_duration,
ROUND(AVG(fare),3) AS average_fare,
COUNT(ride_id) AS rides_count,
ROUND(sum(fare) / NULLIF(sum(duration),0),3) AS fare_per_min,
ROUND(sum(fare) / NULLIF(sum(distance),0),3) AS fare_per_km
FROM t1
GROUP BY age_group
ORDER BY age_group

-- Task 4: Intersectionality between membership status and age group

WITH t1 AS(
SELECT
c.membership_status,
c.age,
CASE
WHEN c.age BETWEEN 18 and 29 THEN '18-29'
WHEN c.age BETWEEN 30 and 44 THEN '30-44'
WHEN c.age BETWEEN 45 and 59 THEN '45-59'
ELSE '60+'
END AS age_group,
r.duration,
r.fare,
r.distance,
r.ride_id
FROM rides r
JOIN customers c
ON r.customer_id = c.customer_id
)
SELECT age_group,
membership_status,
ROUND(AVG(duration),3) AS average_duration,
ROUND(AVG(fare),3) AS average_fare,
COUNT(ride_id) AS rides_count,
ROUND(sum(fare) / NULLIF(sum(duration),0),3) AS fare_per_min,
ROUND(sum(fare) / NULLIF(sum(distance),0),3) AS fare_per_km
FROM t1
GROUP BY age_group, membership_status
ORDER BY age_group

Task 5: Driver Performance Anaylsis

-- 5.1. Top 10 drivers

--Top 10 drivers have been ranked based on a composite score, combining both ride count and average rate
-- This includes normalizing the ride count (dividing ride count by the maximum value of rides per driver) and average rate values. 
-- This involves assigining different weight power for both components based on the business understanding that quality comes over quantity in this regard per se.

WITH t1 AS(
SELECT
d.driver_id,
COUNT(r.ride_id) AS rides_count,
AVG(d.rating) AS average_rating
FROM drivers d
LEFT JOIN rides r
ON d.driver_id = r.driver_id
GROUP BY 1
)

SELECT
driver_id,
ROUND(average_rating,2),
rides_count,
ROUND((0.6*(average_rating/5)) + (0.4*(rides_count/(SELECT MAX(rides_count) FROM t1))),2) * 100 AS composite_score
FROM t1
ORDER BY composite_score DESC
LIMIT 10;

-- 5.2- The key criteria of the driver's quality 
-- This task involves assessing the driver's quality based on tenure (years of experience) and average rating 

-- Attempt 1: Clustering the drivers by their rating into 5 classes 

WITH t1 AS(
SELECT
driver_id,
NTILE(5) OVER(ORDER BY AVG(rating)) AS rating_quntile,
ROUND(AVG(rating),2) AS average_rating
FROM drivers
GROUP BY driver_id
ORDER BY average_rating DESC
)

SELECT t1.rating_quntile,
ROUND(AVG(d.experience_years),2) AS average_experience_years,
PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.experience_years) AS median_experience_years
FROM drivers d
LEFT JOIN t1 ON t1.driver_id = d.driver_id
ORDER BY rating_quntile DESC;

-- Finding: It turns our that there is no actual association between the experience year and rating. In the 5th quintilie representing the top 20% drivers have the least years of experience in comparison to to other groups. Focusing on the following qunitile, the drivers has the same average years of experience as the lowest qunitle. 

Attempt 2: 

WITH t1 AS(
SELECT
driver_id,
rating,
experience_years,
CASE
WHEN rating BETWEEN 1 and 2.49 THEN 'Low Rating'
WHEN rating BETWEEN 2.50 and 3.99 THEN 'Medium Rating'
ELSE 'High Rating'
END AS rating_cateogry
FROM drivers
GROUP BY driver_id, rating, experience_years
ORDER BY rating DESC
)

SELECT rating_cateogry,
count(*) AS drivers_count,
ROUND(AVG(experience_years),2) AS average_experience
FROM t1
GROUP BY rating_cateogry

--Finding: By classifying drivers based on their ratings, reveals that both high- and medium-rated drivers have an average of around 10.3 years of experience. This indicates that years of experience do not play a decisive role in determining drivers' ratings.

5.3. Drivers / Months performance 

SELECT
TO_CHAR(DATE_TRUNC('month',start_time),'YYYY-MM') AS month,
COUNT(ride_id) AS total_rides,
COUNT(DISTINCT driver_id) AS Unique_drivers,
CAST(COUNT(ride_id) / NULLIF(COUNT(DISTINCT driver_id),0) AS FLOAT) AS avg_rides_per_driver
FROM rides
GROUP BY DATE_TRUNC('month',start_time)
ORDER BY month; 

Task 6: Service Quality

6.1. Customer Feedback Analysis: 

WITH t1 AS(
SELECT
f.comments,
f.rating,
r.ride_id,
TO_CHAR(DATE_TRUNC('month',r.start_time),'YYYY-MM') AS month
FROM feedback f
JOIN rides r
ON r.ride_id = f.ride_id
)
,

t2 AS(
SELECT
comments,
month,
ROUND(AVG(rating),3) AS average_rating,
COUNT(t1.*) AS rides_count
FROM t1
GROUP BY comments, month
ORDER BY comments, month
)

SELECT comments, CORR(average_rating, rides_count) AS correlation_coefficient
FROM t2
GROUP BY comments

6.2. Drivers with Low rating due to either late arrival or bad service [Assessing the drivers based on their bad reviewed rides using a metric called 'efficiency rate']

-- Identify the drivers with consistently low ratings and high waiting time
-- Count all the drivers' "Bad reviewed" rides, which are number of rides where the comments given were either 'late arrival' or 'poor service'
-- Come up with a metric to measure the drivers' efficiency rate as follows: "1- (Count of Bad-reviewed rides / total count of rides)".
-- Label the drivers based on their score of the efficiency rate:
-- 1. Efficiency score is less than 0.5 --> 'Needs Improvement'
-- 2. Efficiency Score between 0.5 and 0.649 --> 'Fair Driver'
-- 3. Efficiency Score between 0.65 and 0.799 --> 'Efficient Driver'
-- 4. Efficiency Score more than or equal to 0.8 --> 'Elite Driver'


WITH t1 AS(
SELECT
r.driver_id,
COUNT(CASE WHEN f.comments IN ('late','bad Service') THEN 1 END) AS bad_reviewed_rides,
COUNT(r.*) AS total_rides
FROM feedback f
FULL JOIN rides r
ON r.ride_id = f.ride_id
GROUP BY r.driver_id
ORDER BY total_rides DESC
)

SELECT
driver_id,
ROUND(1- (SUM(bad_reviewed_rides)/SUM(total_rides)),3) AS efficiency_rate,
CASE
WHEN (1- (SUM(bad_reviewed_rides)/SUM(total_rides))) < 0.5 THEN 'Needs Improvement'
WHEN (1- (SUM(bad_reviewed_rides)/SUM(total_rides))) BETWEEN 0.5 AND 0.649 THEN 'Fair Driver'
WHEN (1- (SUM(bad_reviewed_rides)/SUM(total_rides))) BETWEEN 0.65 AND 0.799 THEN 'Efficient Driver'
ELSE 'Elite Driver' END AS efficiency_status
FROM t1
GROUP BY driver_id
ORDER BY efficiency_rate DESC

--Finding: Based on the efficiency score designed, there are 65 efficient or elite drivers; whilst there are 136 drivers who are underperforming.





