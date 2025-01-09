import __init__
import yaml

code_path = __init__.get_code_path()

varaiable_yaml_path = f"{code_path}/config/variables.yaml"

with open(varaiable_yaml_path, "r") as file:
    config = yaml.safe_load(file)


# Subscribe variables
datasets_path = config.get("data_parameters")["datasets_path"]
catalog_name = config.get("data_parameters")["catalog_name"]
bronze_schema_name = config.get("data_parameters")["bronze_schema_name"]
pdf_raw_table_name = config.get("data_parameters")["pdf_raw_table_name"]
