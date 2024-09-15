from pmdd.utils.data_utils import create_db_mp

db_kwargs = {
    "n_samples": 10,
    "res": [32, 32],
}
create_db_mp("magfield", n_workers=10, **db_kwargs)
