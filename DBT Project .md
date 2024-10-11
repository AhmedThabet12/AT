--Create materialized views for the inbound, waste and sales data

CREATE MATERIALIZED VIEW inbound_data_mv AS 
SELECT 
	sku, 
	hub_code,
	SUM(quantity_to - quantity_from) 
FROM inventory.stock_changelogs
WHERE 
	change_reason = "goods_received"
GROUP BY 
	1,2;

CREATE MATERIALIZED VIEW waste_data_mv AS 
SELECT 
	SKU,
	hub_code,
	SUM(quantity_from - quantity_to)
FROM inventory.stock_changelogs
WHERE 
	change_reason IN ('product_expired','product_damaged')
GROUP BY 
	1,2;
CREATE MATERIALIZED VIEW sales_data_mv AS 
SELECT
	SKU, 
	hub_code,
	SUM(quantity_sold)
FROM order_service.order_lineitems
GROUP BY 1,2;

--Check whether the new table name already exists

DO $$
BEGIN 
	-- drop the table if the it exists
	EXECUTE 'DROP TABLE IF EXISTS final_aggregated_data'
	--Rename the old table to another name
	EXECUTE 'ALTER TABLE final_aggregated_data TO old_final_aggregated_data'
END $$;



--Create final aggregated table by left joining the product table with the materialized views


CREATE TABLE final_aggregated_data AS
SELECT
    COALESCE(wd.hub_code, id.hub_code, sd.hub_code) AS hub_code,
    COALESCE(wd.SKU, id.SKU, sd.SKU) AS SKU,
    p.product_name AS product_name,
    p.product_brand AS product_brand,
    p.country_iso AS country,
    COALESCE(sd.total_quantity_sold, 0) AS total_quantity_sold,
    COALESCE(id.total_quantity_inbound, 0) AS total_quantity_inbound,
    COALESCE(wd.total_quantity_waste, 0) AS total_quantity_waste,
    (COALESCE(id.total_quantity_inbound, 0) - COALESCE(sd.total_quantity_sold, 0) - COALESCE(wd.total_quantity_waste, 0)) AS net_quantity
FROM
    products p
LEFT JOIN
    sales_data_mv sd ON p.SKU = sd.SKU
LEFT JOIN
    inbound_data_mv id ON p.SKU = id.SKU
LEFT JOIN
    waste_data_mv wd ON p.SKU = wd.SKU
GROUP BY 
1,2,3,4,5;
