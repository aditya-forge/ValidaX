-- Fraud pattern analysis over the IEEE-CIS Fraud Detection dataset (DuckDB).
--
-- Source: the raw Kaggle tables data/raw/train_transaction.csv and
-- data/raw/train_identity.csv, left-joined on TransactionID. Nothing is written
-- to disk; every object below is an in-memory view or temp table.
--
-- Run from the repository root (the data paths below are relative to it):
--     duckdb -c ".read sql/fraud_analysis.sql"
-- or from Python:
--     import duckdb; con = duckdb.connect(); con.execute(open("sql/fraud_analysis.sql").read())
-- Requires DuckDB >= 1.1 (for *COLUMNS unpacking); tested with DuckDB 2.0.
--
-- TIME: TransactionDT is a number of seconds from an undisclosed reference
-- point, NOT a calendar timestamp. All time buckets below are relative
-- offsets: relative_day = TransactionDT // 86400, relative_hour =
-- (TransactionDT // 3600) % 24, relative_day_of_cycle = relative_day % 7. They
-- are positions in relative cycles, not real dates, hours or weekdays.
--
-- MASKED COLUMNS: C1-C14, D1-D15, M1-M9, V1-V339 and id_* have their meaning
-- withheld by Vesta. Business queries (sections 1-11) use only documented
-- columns. Section 12 summarises masked columns by FAMILY only (data
-- availability per family), with generic labels and no interpretation of any
-- individual column.

-- 0. Sources -------------------------------------------------------------------------
-- M1-M9 are read as text ('T'/'F', and 'M0'/'M1'/'M2' for M4) so the family has
-- one consistent type, matching how pandas loads them in src/data_processing.py.
CREATE OR REPLACE VIEW train_transaction AS
    SELECT * FROM read_csv('data/raw/train_transaction.csv', header = true, types = {
        'M1': 'VARCHAR', 'M2': 'VARCHAR', 'M3': 'VARCHAR', 'M4': 'VARCHAR', 'M5': 'VARCHAR',
        'M6': 'VARCHAR', 'M7': 'VARCHAR', 'M8': 'VARCHAR', 'M9': 'VARCHAR'
    });

CREATE OR REPLACE VIEW train_identity AS
    SELECT * FROM read_csv('data/raw/train_identity.csv', header = true);

-- Full joined table (all 434 columns); only scanned by section 12.
CREATE OR REPLACE VIEW transactions_raw AS
    SELECT t.*, i.* EXCLUDE (TransactionID)
    FROM train_transaction AS t
    LEFT JOIN train_identity AS i USING (TransactionID);

-- Narrow working table: documented columns, relative-time buckets, and one
-- "any value present" flag per masked family. Built in one pass so the
-- business queries don't re-read the CSVs.
CREATE OR REPLACE TEMP TABLE transactions AS
WITH base AS (
    SELECT
        t.TransactionID, t.isFraud, t.TransactionDT, t.TransactionAmt, t.ProductCD,
        t.card4, t.card6, t.P_emaildomain, t.R_emaildomain,
        i.DeviceType, i.DeviceInfo,
        i.TransactionID IS NOT NULL               AS has_identity,
        -- C/D/M/V columns exist only in train_transaction, so these patterns
        -- cannot pick up identity columns (DeviceType/DeviceInfo don't match).
        COALESCE(*COLUMNS('^C\d+$'))              AS c_any,
        COALESCE(*COLUMNS('^D\d+$'))              AS d_any,
        COALESCE(*COLUMNS('^M\d+$'))              AS m_any,
        COALESCE(*COLUMNS('^V\d+$'))              AS v_any
    FROM train_transaction AS t
    LEFT JOIN train_identity AS i USING (TransactionID)
)
SELECT
    * EXCLUDE (c_any, d_any, m_any, v_any),
    TransactionDT // 86400                        AS relative_day,
    (TransactionDT // 3600) % 24                  AS relative_hour,
    (TransactionDT // 86400) % 7                  AS relative_day_of_cycle,
    c_any IS NOT NULL                             AS c_family_present,
    d_any IS NOT NULL                             AS d_family_present,
    m_any IS NOT NULL                             AS m_family_present,
    v_any IS NOT NULL                             AS v_family_present
FROM base;

-- 1. Overall volume and fraud rate ----------------------------------------------------
SELECT
    COUNT(*)                                                   AS total_transactions,
    SUM(isFraud)                                               AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)                               AS fraud_rate_pct,
    ROUND(SUM(TransactionAmt * isFraud), 2)                    AS total_fraud_value_usd,
    ROUND(AVG(TransactionAmt) FILTER (WHERE isFraud = 1), 2)    AS avg_fraud_amount,
    ROUND(MEDIAN(TransactionAmt) FILTER (WHERE isFraud = 1), 2) AS median_fraud_amount,
    ROUND(AVG(TransactionAmt) FILTER (WHERE isFraud = 0), 2)    AS avg_legit_amount,
    ROUND(AVG(has_identity::INTEGER) * 100, 2)                 AS identity_match_pct
FROM transactions;

-- 2. Chronological 80/20 split check ---------------------------------------------------
-- Same rule as src/model.py: order by TransactionDT (TransactionID breaks ties);
-- the first floor(0.8 * n) rows are train, the rest test.
-- Expected on the full dataset: 472,432 train / 118,108 test.
WITH ordered AS (
    SELECT
        TransactionDT, relative_day, isFraud,
        ROW_NUMBER() OVER (ORDER BY TransactionDT, TransactionID) AS row_num,
        COUNT(*) OVER ()                                          AS n_rows
    FROM transactions
)
SELECT
    CASE WHEN row_num <= FLOOR(0.8 * n_rows) THEN 'train (first 80%)' ELSE 'test (last 20%)' END
                                   AS segment,
    COUNT(*)                       AS transactions,
    MIN(TransactionDT)             AS min_offset_seconds,
    MAX(TransactionDT)             AS max_offset_seconds,
    MIN(relative_day)              AS first_relative_day,
    MAX(relative_day)              AS last_relative_day,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM ordered
GROUP BY 1
ORDER BY MIN(row_num);

-- 3. Volume and fraud rate by relative day (ordered by offset) ------------------------
SELECT
    relative_day,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY relative_day
ORDER BY relative_day;

-- 4. Fraud rate by relative hour (position in the relative 24-hour cycle) -------------
SELECT
    relative_hour,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY relative_hour
ORDER BY relative_hour;

-- 5. Fraud rate by relative day-of-cycle (position in the relative 7-day cycle) -------
SELECT
    relative_day_of_cycle,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY relative_day_of_cycle
ORDER BY relative_day_of_cycle;

-- 6. Fraud rate by product code --------------------------------------------------------
SELECT
    ProductCD,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2)  AS avg_amount
FROM transactions
GROUP BY ProductCD
ORDER BY fraud_rate_pct DESC;

-- 7. Fraud rate by card network / card type --------------------------------------------
SELECT
    COALESCE(card4, 'unknown')     AS card_network,
    COALESCE(card6, 'unknown')     AS card_type,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY 1, 2
ORDER BY fraud_rate_pct DESC;

-- 8. Fraud rate by transaction amount band ---------------------------------------------
SELECT
    CASE
        WHEN TransactionAmt < 25   THEN '<25'
        WHEN TransactionAmt < 100  THEN '25-99'
        WHEN TransactionAmt < 500  THEN '100-499'
        WHEN TransactionAmt < 2000 THEN '500-1999'
        ELSE '2000+'
    END                            AS amount_band,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY MIN(TransactionAmt);

-- 9. Fraud rate by purchaser email domain (top 20 by volume) ---------------------------
SELECT
    COALESCE(P_emaildomain, 'missing') AS purchaser_email_domain,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY transactions DESC
LIMIT 20;

-- 10. Identity/device signals: presence and device type --------------------------------
SELECT
    CASE WHEN NOT has_identity THEN 'no identity data'
         ELSE COALESCE(DeviceType, 'identity data, device type missing') END AS device_type,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;

-- 11a. High-value transactions (top 1% by amount): share of all fraud ------------------
WITH ranked AS (
    SELECT TransactionAmt, isFraud,
           PERCENT_RANK() OVER (ORDER BY TransactionAmt) AS pct_rank
    FROM transactions
)
SELECT
    COUNT(*)                       AS top_1pct_transactions,
    SUM(isFraud)                   AS fraud_in_top_1pct,
    ROUND(100.0 * SUM(isFraud) / (SELECT SUM(isFraud) FROM transactions), 2)
                                   AS pct_of_all_fraud
FROM ranked
WHERE pct_rank >= 0.99;

-- 11b. Purchaser vs. recipient email domain ---------------------------------------------
SELECT
    CASE
        WHEN R_emaildomain IS NULL         THEN 'no recipient email'
        WHEN P_emaildomain = R_emaildomain THEN 'same domain'
        ELSE 'different domain'
    END                            AS email_match_status,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;

-- 12a. Masked families: data availability (generic family labels only) ----------------
-- Non-null counts per masked column, aggregated to family level. Describes how
-- much data each family holds, never what any column means.
WITH counts AS (
    SELECT COUNT(*) AS n_rows, COUNT(COLUMNS('^([CDMV]\d+|id_\d+)$'))
    FROM transactions_raw
),
per_column AS (
    UNPIVOT counts ON COLUMNS(* EXCLUDE (n_rows)) INTO NAME column_name VALUE non_null
)
SELECT
    CASE regexp_extract(column_name, '^(id_|[CDMV])')
        WHEN 'C'   THEN 'masked counting features (C1-C14)'
        WHEN 'D'   THEN 'masked time-delta features (D1-D15)'
        WHEN 'M'   THEN 'masked match flags (M1-M9)'
        WHEN 'V'   THEN 'masked Vesta-engineered features (V1-V339)'
        WHEN 'id_' THEN 'masked identity/device signals (id_*)'
    END                                              AS masked_family,
    COUNT(*)                                         AS columns,
    ROUND(100 * (1 - AVG(non_null / n_rows)), 1)     AS avg_missing_pct,
    ROUND(100 * (1 - MAX(non_null / n_rows)), 1)     AS min_missing_pct,
    ROUND(100 * (1 - MIN(non_null / n_rows)), 1)     AS max_missing_pct
FROM per_column
GROUP BY 1
ORDER BY 1;

-- 12b. Masked families: fraud rate when the family has any value vs. none -------------
-- A data-availability signal per family (the id_* family is present exactly
-- when the transaction has an identity record).
WITH presence AS (
    UNPIVOT (
        SELECT isFraud,
               c_family_present AS "masked counting features (C1-C14)",
               d_family_present AS "masked time-delta features (D1-D15)",
               m_family_present AS "masked match flags (M1-M9)",
               v_family_present AS "masked Vesta-engineered features (V1-V339)",
               has_identity     AS "masked identity/device signals (id_*)"
        FROM transactions
    ) ON COLUMNS(* EXCLUDE (isFraud)) INTO NAME masked_family VALUE any_value_present
)
SELECT
    masked_family,
    any_value_present,
    COUNT(*)                       AS transactions,
    SUM(isFraud)                   AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)   AS fraud_rate_pct
FROM presence
GROUP BY 1, 2
ORDER BY 1, 2 DESC;
