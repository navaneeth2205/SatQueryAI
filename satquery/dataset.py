"""
Query Dataset Loader and Preprocessing for SatQuery AI Task Router.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from satquery.taxonomy import TaskType, InputType


REQUIRED_COLUMNS = {"query", "task", "input_type"}


def load_queries_dataset(
    csv_path: str | Path,
    val_split_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate query dataset. If 'split' column exists, uses it;
    otherwise performs stratified train/validation split.
    
    Returns:
        (train_df, val_df)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Clean queries
    df["query"] = df["query"].astype(str).str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["input_type"] = df["input_type"].astype(str).str.strip()

    # Drop empty queries
    df = df[df["query"].str.len() > 0].copy()

    # Validate task types
    valid_tasks = {t.value for t in TaskType}
    invalid_tasks = set(df["task"].unique()) - valid_tasks
    if invalid_tasks:
        raise ValueError(f"Encountered unknown task labels in dataset: {invalid_tasks}")

    # Use explicit split column if present
    if "split" in df.columns:
        train_df = df[df["split"].str.lower() == "train"].copy().reset_index(drop=True)
        val_df = df[df["split"].str.lower().isin(["val", "validation", "test"])].copy().reset_index(drop=True)
        
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("Split column present but train or val subset is empty.")
        return train_df, val_df

    # Otherwise perform stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=val_split_ratio,
        random_state=random_state,
        stratify=df["task"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate summary statistics for a dataset split."""
    return {
        "total_samples": len(df),
        "task_counts": df["task"].value_counts().to_dict(),
        "input_type_counts": df["input_type"].value_counts().to_dict(),
        "avg_query_word_count": round(float(df["query"].apply(lambda q: len(q.split())).mean()), 2)
    }
