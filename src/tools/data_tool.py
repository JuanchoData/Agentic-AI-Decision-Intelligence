from pathlib import Path

import duckdb
import pandas as pd


def query_csv(
    csv_path: str,
    sql_query: str,
) -> pd.DataFrame:
    """
    Run a SQL query against a CSV file using DuckDB.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    sql_query : str
        SQL query to execute.
        The CSV is exposed as a DuckDB view named 'data'.

    Returns
    -------
    pandas.DataFrame
        Query results as a pandas DataFrame.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # Convert Windows paths to a DuckDB-friendly format.
    safe_path = (
        str(path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW data AS
            SELECT *
            FROM read_csv_auto('{safe_path}');
            """
        )

        result = connection.execute(
            sql_query
        ).fetchdf()

        return result

    finally:
        connection.close()


if __name__ == "__main__":
    print("DuckDB data tool ready.")