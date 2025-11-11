"""
etl.py
Load the Olist CSVs, perform basic cleaning, joins, and write denormalized parquet tables.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Hardcoded paths
DATA_DIR = "./data/ecommerce"
PARQUET_OUT = "./data/parquet"

def load_csvs(data_dir):
    mapping = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "cat_trans": "product_category_name_translation.csv"
    }
    dfs = {}
    for k, fname in mapping.items():
        path = Path(data_dir) / fname
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            dfs[k] = None
        else:
            if k == "orders":
                dfs[k] = pd.read_csv(path, parse_dates=[
                    'order_purchase_timestamp',
                    'order_approved_at',
                    'order_delivered_customer_date',
                    'order_estimated_delivery_date'
                ])
            else:
                dfs[k] = pd.read_csv(path)
            print(f"Loaded {k}: {dfs[k].shape}")
    return dfs


def build_denorm(dfs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    orders = dfs['orders']
    items = dfs['items']
    payments = dfs['payments']
    products = dfs['products']
    customers = dfs['customers']
    reviews = dfs['reviews']

    payments_agg = payments.groupby('order_id', as_index=False).agg({
        'payment_type': lambda x: x.mode().iat[0] if len(x) > 0 else None,
        'payment_installments': 'max',
        'payment_value': 'sum'
    })

    items = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    items = items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp',
                                'order_status', 'order_delivered_customer_date',
                                'order_estimated_delivery_date']], on='order_id', how='left')
    items = items.merge(payments_agg[['order_id', 'payment_value', 'payment_type']], on='order_id', how='left')
    items = items.merge(customers[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']],
                        on='customer_id', how='left')

    items['order_purchase_timestamp'] = pd.to_datetime(items['order_purchase_timestamp'])
    items['order_delivered_customer_date'] = pd.to_datetime(items['order_delivered_customer_date'])
    items['delivery_days'] = (items['order_delivered_customer_date'] - items['order_purchase_timestamp']).dt.days
    items['order_year_month'] = items['order_purchase_timestamp'].dt.to_period('M').astype(str)
    items['gmv'] = items['price']

    # Save Parquet files
    items.to_parquet(os.path.join(out_dir, 'order_items_denorm.parquet'), index=False)
    orders.to_parquet(os.path.join(out_dir, 'orders.parquet'), index=False)
    customers.to_parquet(os.path.join(out_dir, 'customers.parquet'), index=False)
    if reviews is not None:
        reviews.to_parquet(os.path.join(out_dir, 'reviews.parquet'), index=False)

    # KPIs
    kpi_weekly = items.set_index('order_purchase_timestamp').resample('W').agg({
        'gmv': 'sum',
        'delivery_days': 'mean',
        'order_id': 'nunique'
    }).rename(columns={'order_id': 'num_orders'}).reset_index()
    kpi_weekly.to_parquet(os.path.join(out_dir, 'kpi_weekly.parquet'), index=False)

    # RFM
    rfmdf = items.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'gmv': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'last_purchase',
                       'order_id': 'frequency',
                       'gmv': 'monetary'}).reset_index()
    now = items['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfmdf['recency_days'] = (now - rfmdf['last_purchase']).dt.days
    rfmdf['r_score'] = pd.qcut(rfmdf['recency_days'], 5, labels=[5,4,3,2,1]).astype(int)
    rfmdf['f_score'] = pd.qcut(rfmdf['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfmdf['m_score'] = pd.qcut(rfmdf['monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfmdf['rfm_score'] = rfmdf['r_score']*100 + rfmdf['f_score']*10 + rfmdf['m_score']
    rfmdf.to_parquet(os.path.join(out_dir, 'rfm.parquet'), index=False)

    print("✅ ETL complete. Parquet saved at:", out_dir)


if __name__ == "__main__":
    dfs = load_csvs(DATA_DIR)
    build_denorm(dfs, PARQUET_OUT)
