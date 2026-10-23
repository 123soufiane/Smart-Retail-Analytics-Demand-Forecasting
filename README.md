# Smart-Retail-Analytics-Demand-Forecasting

## Database schema

Install the project dependencies, configure `DATABASE_URL` in the repository's
`.env` file, then run this command from the repository root:

```bash
python -m src.database.create_schema
```

The command creates or updates the tables used by the loaders: categories,
suppliers, products, stores, customers, inventory, sales, sale items, returns,
supplier orders, and purchase items. It is safe to rerun; existing tables are
updated in place without dropping existing data.

Data is then loaded with the ingestion scripts:

```bash
python -m src.data_ingestion.insert_uci_sales_fast      # UCI Online Retail
python -m src.data_ingestion.load_store_item_to_neon     # Store Item
```

## Loading Store Item sales into Neon

Place the Store Item dataset at
`data/raw/store_item/retail_sales.csv`, then run:

```bash
python -m src.data_ingestion.load_store_item_to_neon
```

The loader maps each source store and item to `stores` and `products`, and
imports each store/item/day row into `sales` and `sale_items`. It is built for
large files and for Neon's storage limit:

- Rows are written with PostgreSQL `COPY` in one transaction per source store,
  so a store is either fully loaded or not loaded at all.
- Stores that are already complete are skipped, and a partially loaded store is
  cleared and reloaded. Rerunning the command never duplicates rows.
- Before each store the loader checks `pg_database_size` and stops cleanly when
  the next store would not fit. The threshold defaults to 990 MB; override it
  with `STORE_ITEM_SIZE_LIMIT_MB` when the Neon project has more storage.
- To continue after freeing space, simply run the same command again.

On a 1 GB Neon project the normalized `sales` + `sale_items` layout fits 42 of
the 50 stores (~3.8M rows). All 50 stores in this layout need roughly
1.25-1.3 GB; use a larger Neon plan, or free space, to load the rest.

To make room for the extra stores, three secondary indexes were dropped:
`sales_sale_date_idx`, `sales_store_date_idx`, and `sale_items_product_idx`.
Recreate them after upgrading the Neon plan if analytic queries need them.

The source does not include product costs, so the loader initializes each
product's catalog cost from its average source price. Replace those values
with actual costs before using the database for margin analysis.

## Running the pipeline

```bash
# Default: train the weekly Store Item model and predict next week
python -m src.pipeline.run_full_pipeline

# One model per product + store series (2,100 rows for 42 stores x 50 items)
python -m src.pipeline.run_full_pipeline --by-store

# Also run EDA and/or rebuild the daily feature file (both are slower)
python -m src.pipeline.run_full_pipeline --eda
python -m src.pipeline.run_full_pipeline --features
```

Individual stages:

```bash
python -m src.analytics.eda                              # EDA -> reports/eda/*.png
python -m src.features.feature_engineering               # data/features_demand.parquet
python -m src.models.train_strong SITEM-                 # product-level model
python -m src.models.train_strong SITEM- --by-store      # per-store model
python -m src.models.predict --top 10                    # product-level predictions
python -m src.models.predict_store --top 15              # per-store predictions
```
