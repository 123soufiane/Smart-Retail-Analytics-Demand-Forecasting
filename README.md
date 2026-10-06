# Smart-Retail-Analytics-Demand-Forecasting

## Database schema

Install the project dependencies, configure `DATABASE_URL` in the repository's
`.env` file, then run this command from the repository root:

```bash
python -m src.database.create_schema
```

The command creates or updates the tables used by the data generator:
categories, suppliers, products, stores, customers, inventory, sales,
sale items, returns, supplier orders, and purchase items. It is safe to rerun;
the earlier starter tables are updated in place without dropping existing data.
After schema setup, generate sample data with:

```bash
python -m src.data_generation.generate_data
```