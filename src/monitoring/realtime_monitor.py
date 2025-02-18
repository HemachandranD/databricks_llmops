from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries, MonitorInfoStatus
from src.config.configuration import catalog_name, gold_schema_name, app_inference_processed_table_name

# Create monitor using databricks-sdk's `quality_monitors` client

def create_lhm_monitor(catalog_name, schema_name, app_inference_processed_table_name):
    try:
        w = WorkspaceClient()
        lhm_monitor = w.quality_monitors.create(
            table_name=app_inference_processed_table_name,
            time_series = MonitorTimeSeries(
            timestamp_col = "timestamp",
            granularities = ["5 minutes"],
            ),
            assets_dir = os.getcwd(),
            slicing_exprs = ["model_id"],
            output_schema_name=f"{catalog_name}.{schema_name}"
        )

        monitor_info = w.quality_monitors.get(app_inference_processed_table_name)

        if monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
            print("Wait until monitor creation is completed...")
        
    except Exception as lhm_exception:
    print(lhm_exception)


# if __name__ == "__main__":
#     app_inference_processed_table_name = f"{catalog_name}.{schema_name}.{app_inference_processed_table_name}
#     create_lhm_monitor(catalog_name, schema_name, app_inference_processed_table_name)
