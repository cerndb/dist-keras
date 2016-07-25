from distkeras import spark_model

model = SparkModel(None, None)
print(model.master_address)
print(model.master_port)
