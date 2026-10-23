"""End-to-end pipeline for the current Neon setup.

Steps:
    1. EDA                              (optional, --eda)
    2. Daily feature rebuild            (optional, --features)
    3. Weekly model training            (Store Item data)
    4. Next-week predictions

Examples:
    python -m src.pipeline.run_full_pipeline
    python -m src.pipeline.run_full_pipeline --eda
    python -m src.pipeline.run_full_pipeline --by-store
    python -m src.pipeline.run_full_pipeline --features --eda
"""

import argparse
from datetime import datetime


def log(message: str):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")


def run_full_pipeline(
    run_eda: bool = False,
    rebuild_features: bool = False,
    by_store: bool = False,
):
    start = datetime.now()
    log("🚀 PIPELINE STARTED")
    print("=" * 65)

    if run_eda:
        log("STEP 1 → EDA")
        from src.analytics.eda import run_full_eda

        run_full_eda()
    else:
        log("STEP 1 → EDA skipped")

    if rebuild_features:
        log("STEP 2 → Feature engineering")
        from src.features.feature_engineering import (
            create_final_features,
            save_features,
        )

        features = create_final_features()
        save_features(features)
        del features
    else:
        log("STEP 2 → Feature engineering skipped")

    log("STEP 3 → Training weekly model")
    from src.models.train_strong import train_strong

    results = train_strong("SITEM-", by_store=by_store)

    log("STEP 4 → Predictions")
    if by_store:
        from src.models.predict_store import predict_all

        predictions = predict_all()
        print(predictions.head(10).to_string(index=False))
    else:
        from src.models.predict import predict_top

        predictions = predict_top(10)

    log(f"🎉 PIPELINE FINISHED in {datetime.now() - start}")
    print("=" * 65)
    return results, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", action="store_true", help="run EDA")
    parser.add_argument("--features", action="store_true",
                        help="rebuild data/features_demand.parquet (slow)")
    parser.add_argument("--by-store", action="store_true",
                        help="train/predict per product+store series")
    args = parser.parse_args()

    run_full_pipeline(
        run_eda=args.eda,
        rebuild_features=args.features,
        by_store=args.by_store,
    )
