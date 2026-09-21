from src.tools.data_tool import query_csv


DEFAULT_DATA_PATH = "data/raw/operational_data.csv"


def summarize_operations(
    csv_path: str = DEFAULT_DATA_PATH,
):
    """
    Return a high-level summary of the operational dataset.
    """

    sql_query = """
    SELECT
        COUNT(*) AS n_rows,
        ROUND(AVG(quality_score), 2) AS avg_quality,
        ROUND(AVG(temperature), 2) AS avg_temperature,
        ROUND(AVG(pressure), 2) AS avg_pressure,
        ROUND(AVG(throughput), 2) AS avg_throughput,
        ROUND(AVG(downtime_minutes), 2) AS avg_downtime,
        SUM(
            CASE
                WHEN status = 'attention' THEN 1
                ELSE 0
            END
        ) AS attention_count
    FROM data
    """

    return query_csv(csv_path, sql_query)


def compare_status_groups(
    csv_path: str = DEFAULT_DATA_PATH,
):
    """
    Compare operational metrics between status groups.
    """

    sql_query = """
    SELECT
        status,
        COUNT(*) AS n_records,
        ROUND(AVG(quality_score), 2) AS avg_quality,
        ROUND(AVG(temperature), 2) AS avg_temperature,
        ROUND(AVG(pressure), 2) AS avg_pressure,
        ROUND(AVG(throughput), 2) AS avg_throughput,
        ROUND(AVG(downtime_minutes), 2) AS avg_downtime
    FROM data
    GROUP BY status
    ORDER BY avg_quality DESC
    """

    return query_csv(csv_path, sql_query)


def find_low_quality_cases(
    csv_path: str = DEFAULT_DATA_PATH,
    limit: int = 10,
):
    """
    Return the observations with the lowest quality scores.
    """

    sql_query = f"""
    SELECT
        timestamp,
        temperature,
        pressure,
        throughput,
        downtime_minutes,
        quality_score,
        status
    FROM data
    ORDER BY quality_score ASC
    LIMIT {int(limit)}
    """

    return query_csv(csv_path, sql_query)


def quality_correlations(
    csv_path: str = DEFAULT_DATA_PATH,
):
    """
    Calculate correlations between operational variables
    and quality score.
    """

    sql_query = """
    SELECT
        ROUND(CORR(temperature, quality_score), 3)
            AS temperature_correlation,

        ROUND(CORR(pressure, quality_score), 3)
            AS pressure_correlation,

        ROUND(CORR(throughput, quality_score), 3)
            AS throughput_correlation,

        ROUND(CORR(downtime_minutes, quality_score), 3)
            AS downtime_correlation
    FROM data
    """

    return query_csv(csv_path, sql_query)


if __name__ == "__main__":

    print("\n=== OPERATIONAL SUMMARY ===")
    print(summarize_operations())

    print("\n=== STATUS COMPARISON ===")
    print(compare_status_groups())

    print("\n=== LOW QUALITY CASES ===")
    print(find_low_quality_cases(limit=5))

    print("\n=== QUALITY CORRELATIONS ===")
    print(quality_correlations())