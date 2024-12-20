SQL Queries Used
CREATING OR VIEWING FORESTATION TABLE:
CREATE VIEW forestation2 AS
SELECT f.country_code AS forest_country_code, f.country_name AS forest_country_name, f.year AS forest_year, f.forest_area_sqkm AS forest_area_designated, 
l.country_code AS land_country_code, l.country_name AS land_country_name, l.year AS land_year, l.total_area_sq_mi * 2.59 AS land_area_km, 
r.country_name AS region_country_name, r.country_code AS region_country_code, r.region AS region_name, r. income_group AS region_income_group,
(f.forest_area_sqkm/(l.total_area_sq_mi * 2.59)) AS land_area_percent_as_forest
FROM forest_area f 
JOIN land_area l 
ON f.year = l.year
JOIN regions r
ON l.country_code = r.country_code
WHERE f.year = l.year;

CREATE VIEW forestation3 AS
SELECT f.country_code AS forest_country_code, f.country_name AS forest_country_name, f.year AS forest_year, f.forest_area_sqkm AS forest_area_designated, 
l.country_code AS land_country_code, l.country_name AS land_country_name, l.year AS land_year, l.total_area_sq_mi * 2.59 AS land_area_km, 
r.country_name AS region_country_name, r.country_code AS region_country_code, r.region AS region_name, r. income_group AS region_income_group,
(sum(f.forest_area_sqkm)/sum(l.total_area_sq_mi * 2.59)*100) AS land_area_percent_as_forest
FROM forest_area f 
JOIN land_area l 
ON f.year = l.year
JOIN regions r
ON l.country_code = r.country_code
WHERE f.year = l.year;

_________________________________________________

World Outlook 

SELECT sum(forest_area_sqkm) AS total_forest_area_1990
FROM forest_area
WHERE year = '1990' AND country_name = 'World';
_________________________________________________

SELECT sum(forest_area_sqkm) AS total_forest_area_2016
FROM forest_area
WHERE year = '2016' AND country_name = 'World';

_________________________________________________

WITH forest_2016 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_2016
	FROM forest_area
	WHERE year = '2016' AND country_name = 'World'
	GROUP BY 1
),
forest_1990 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_1990
	FROM forest_area
	WHERE year = '1990' AND country_name = 'World'
  	GROUP BY 1
)
SELECT forest_1990.total_forest_area_1990 - forest_2016.total_forest_area_2016 AS forest_difference
FROM forest_2016
JOIN forest_1990
ON forest_1990.country_code = forest_2016.country_code;
____________________________________________________

WITH forest_2016 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_2016
	FROM forest_area
	WHERE year = '2016' AND country_name = 'World'
	GROUP BY 1
),
forest_1990 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_1990
	FROM forest_area
	WHERE year = '1990' AND country_name = 'World'
  	GROUP BY 1
)
SELECT ((forest_1990.total_forest_area_1990 - forest_2016.total_forest_area_2016) / forest_2016.total_forest_area_2016)*100 AS forest_decreasing_rate
FROM forest_2016
JOIN forest_1990
ON forest_1990.country_code = forest_2016.country_code;
________________________________________________________
SELECT land_country_name, sum(land_area_km)
FROM forestation2
WHERE land_year = '2016'
AND land_area_km < (WITH forest_2016 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_2016
	FROM forest_area
	WHERE year = '2016' AND country_name = 'World'
	GROUP BY 1
),
forest_1990 AS (
	SELECT country_code, sum(forest_area_sqkm) AS total_forest_area_1990
	FROM forest_area
	WHERE year = '1990' AND country_name = 'World'
  	GROUP BY 1
)
SELECT forest_1990.total_forest_area_1990 - forest_2016.total_forest_area_2016 AS forest_difference
FROM forest_2016
JOIN forest_1990
ON forest_1990.country_code = forest_2016.country_code
)
GROUP BY 1
LIMIT 1;
______________________________________________________
2. Regional Outlook:

SELECT forest_country_name, sum(forest_area_designated) / SUM (land_area_km) AS forest_land_ratio
FROM forestation2
WHERE forest_year = '2016'
AND forest_country_name = 'World'
GROUP BY 1;

___________________________________________________________


SELECT f.year, r.region, ((SUM(f.forest_area_sqkm)/SUM(l.total_area_sq_mi)*2.59)) AS forestation_rate
FROM forest_area f
JOIN land_area l
ON f.country_code = l.country_code AND f.year = l.year
JOIN regions r
ON r.country_code = l.country_code
WHERE f.year = '2016'
GROUP BY 1,2
ORDER BY 3 DESC
LIMIT 1;
_________________________________________________________

SELECT f.year, r.region, ((SUM(f.forest_area_sqkm)/SUM(l.total_area_sq_mi)*2.59)) AS forestation_rate
FROM forest_area f
JOIN land_area l
ON f.country_code = l.country_code AND f.year = l.year
JOIN regions r
ON r.country_code = l.country_code
WHERE f.year = '2016'
GROUP BY 1,2
ORDER BY 3
LIMIT 1;
_______________________________________________
SELECT forest_country_name, sum(forest_area_designated) / SUM (land_area_km) AS forest_land_ratio
FROM forestation2
WHERE forest_year = '2016'
AND forest_country_name = 'World'
GROUP BY 1;
______________________________________________________
SELECT forest_country_name, sum(forest_area_designated) / SUM (land_area_km) AS forest_land_ratio
FROM forestation2
WHERE forest_year = '1990'
AND forest_country_name = 'World'
GROUP BY 1;

________________________________________________

SELECT f.year, r.region, ((SUM(f.forest_area_sqkm)/SUM(l.total_area_sq_mi)*2.59)) AS forestation_rate
FROM forest_area f
JOIN land_area l
ON f.country_code = l.country_code AND f.year = l.year
JOIN regions r
ON r.country_code = l.country_code
WHERE f.year = '1990'
GROUP BY 1,2
ORDER BY 3 DESC
LIMIT 1;

___________________________________________________

WITH t1 AS (
SELECT  r.region, ((SUM(f.forest_area_sqkm)/SUM(l.total_area_sq_mi)*2.59)) AS forestation_rate_2016
FROM forest_area f
JOIN land_area l
ON f.country_code = l.country_code AND f.year = l.year
JOIN regions r
ON r.country_code = l.country_code
WHERE f.year = '2016'
GROUP BY 1
ORDER BY 2
),

t2 AS(
SELECT  r.region, ((SUM(f.forest_area_sqkm)/SUM(l.total_area_sq_mi)*2.59)) AS forestation_rate_1990
FROM forest_area f
JOIN land_area l
ON f.country_code = l.country_code AND f.year = l.year
JOIN regions r
ON r.country_code = l.country_code
WHERE f.year = '1990'
GROUP BY 1
ORDER BY 2
)
SELECT t1.region, t1.forestation_rate_2016 - t2.forestation_rate_1990 AS forestation_difference
FROM t1
JOIN t2
ON t1.region = t2.region
GROUP BY 1,2;

____________________________________________________________________________


  WITH t1 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_2016
  FROM forest_area f
  WHERE f.year = '2016'
), 
t2 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_1990
  FROM forest_area f
  WHERE f.year = '1990'
)
SELECT t1.country_name, t1.forest_2016, t2.forest_1990, MAX(t1.forest_2016 - t2.forest_1990) AS best_forest
FROM t1
JOIN t2
ON t1.country_name = t2.country_name
WHERE t1.forest_2016 IS NOT NULL
AND t2.forest_1990 IS NOT NULL
GROUP BY 1,2,3
ORDER BY 4 DESC
LIMIT 1;
______________________________________________

WITH t1 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_2016
  FROM forest_area f
  WHERE f.year = '2016'
), 
t2 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_1990
  FROM forest_area f
  WHERE f.year = '1990'
)
SELECT t1.country_name, t1.forest_2016, t2.forest_1990, MAX(t1.forest_2016 - t2.forest_1990) AS best_forest
FROM t1
JOIN t2
ON t1.country_name = t2.country_name
WHERE t1.forest_2016 IS NOT NULL
AND t2.forest_1990 IS NOT NULL
GROUP BY 1,2,3
ORDER BY 4 DESC
LIMIT 2;

___________________________________________________

WITH t1 AS (
  SELECT f.country_name, f.forest_area_sqkm/(l.total_area_sq_mi*2.59) AS forest_to_land_2016
  FROM forest_area f
  JOIN land_area l
  ON f.country_name = l.country_name 
  AND f.year = l.year
  WHERE f.year = '2016'
),
t2 AS (
  SELECT f.country_name, f.forest_area_sqkm/(l.total_area_sq_mi*2.59) AS forest_to_land_1990
  FROM forest_area f
  JOIN land_area l
  ON f.country_name = l.country_name 
  AND f.year = l.year
  WHERE f.year = '1990'
)
SELECT t1.country_name, t1.forest_to_land_2016, t2.forest_to_land_1990, ((t1.forest_to_land_2016 - t2.forest_to_land_1990)/t2.forest_to_land_1990) AS best_forest
FROM t1
JOIN t2
ON t1.country_name = t2.country_name
WHERE forest_to_land_2016 IS NOT NULL
AND t2.forest_to_land_1990 IS NOT NULL
GROUP BY 1,2,3
ORDER BY 4 DESC
LIMIT 2;
__________________________________________________________

WITH t1 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_2016
  FROM forest_area f
  WHERE f.year = '2016'
), 
t2 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_1990
  FROM forest_area f
  WHERE f.year = '1990'
)
SELECT t1.country_name, r.region, t1.forest_2016, t2.forest_1990, (t1.forest_2016 - t2.forest_1990) AS worst_forestation
FROM t1
JOIN t2
ON t1.country_name = t2.country_name
JOIN regions r
ON r.country_name = t2.country_name
WHERE t1.forest_2016 IS NOT NULL
AND t2.forest_1990 IS NOT NULL
AND t1.country_name != 'World'
GROUP BY 1,2,3,4
ORDER BY 5
LIMIT 5;

_________________________________________

WITH t1 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_2016
  FROM forest_area f
  WHERE f.year = '2016'
), 
t2 AS (
  SELECT f.country_name, f.forest_area_sqkm AS forest_1990
  FROM forest_area f
  WHERE f.year = '1990'
)
SELECT t1.country_name, r.region, t1.forest_2016, t2.forest_1990, (((t1.forest_2016 - t2.forest_1990)/t2.forest_1990)*100) AS worst_forestation_percent
FROM t1
JOIN t2
ON t1.country_name = t2.country_name
JOIN regions r
ON r.country_name = t2.country_name
WHERE t1.forest_2016 IS NOT NULL
AND t2.forest_1990 IS NOT NULL
AND t1.country_name != 'World'
GROUP BY 1,2,3,4
ORDER BY 5
LIMIT 5;

_____________________________________________________________
 Quartiles:

SELECT quartiles, Count(*)
            FROM (
                        SELECT t1.country_name,  
                        CASE WHEN t1.forestation_percent <= 25 THEN '0-25%' 
                        WHEN t1.forestation_percent > 25 AND t1.forestation_percent <= 50 THEN '25-50%'  
                        WHEN t1.forestation_percent > 50 AND t1.forestation_percent <= 75 THEN '50-75%' 
                        WHEN t1.forestation_percent > 75 AND t1.forestation_percent <= 100 THEN '75-100%' 
                        END quartiles 
                        FROM (
                              SELECT f.country_name, (sum(f.forest_area_sqkm) / sum(l.total_area_sq_mi * 2.59)*100) AS forestation_percent
                              FROM forest_area f
                              JOIN land_area l
                              ON f.year = l.year
                              WHERE f.year = 2016
                              GROUP BY 1)t1
                  )t2
            GROUP BY 1
            ORDER BY 1

_____________________________________________________________

SELECT t1.country_name,r.region,t1.forestation_percent

FROM (
      SELECT l.country_name, (sum(f.forest_area_sqkm) / sum(l.total_area_sq_mi * 2.59)*100) AS forestation_percent
      FROM forest_area f
      JOIN land_area l
      ON f.country_name = l.country_name
      WHERE l.year = 2016
      GROUP BY 1
      ORDER BY 2
      )t1
JOIN regions r
ON t1.country_name = r.country_name
WHERE t1.forestation_percent BETWEEN 25 AND 50 
AND t1.country_name != 'World'
GROUP BY 1,2,3
ORDER BY 2 DESC