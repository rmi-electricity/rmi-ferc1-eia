"""Load EQR contracts and identities to a sqlite database."""
import argparse
import io
import logging
import shutil
import sys
import zipfile
from pathlib import Path
import multiprocessing as mp
from typing import NamedTuple
import numpy as np
import coloredlogs
import pandas as pd
import sqlalchemy as sa
from tqdm import tqdm

from pudl_rmi import EQR_DATA_DIR, EQR_DB_PATH, OUTPUTS_DIR

logger = logging.getLogger(__name__)

engine = sa.create_engine(f"sqlite:///{EQR_DB_PATH}")

FILE_END_STRS_TO_TABLE_NAMES = {
    # "indexPub.CSV": "index_publishing",
    # "ident.CSV": "identities",
    # "contracts.CSV": "contracts",
    "transactions.CSV": "transactions",
}

TABLE_DTYPES = {
    "identities": {"contact_zip": "string", "contact_phone": "string"},
    "contracts": {
        "seller_history_name": "string",
        "increment_name": str,
        "point_of_receipt_specific_location": str,
        "point_of_delivery_specific_location": str,
    },
    "transactions": {
        "transaction_unique_id": "string",
        "seller_company_name": "string",
        "customer_company_name": "string",
        "ferc_tariff_reference": "string",
        "contract_service_agreement": "string",
        "transaction_unique_identifier": "string",
        "exchange_brokerage_service": "string",
        "type_of_rate": "string",
        "time_zone": "string",
        "point_of_delivery_balancing_authority": "string",
        "point_of_delivery_specific_location": "string",
        "class_name": "string",
        "term_name": "string",
        "increment_name": "string",
        "increment_peaking_name": "string",
        "product_name": "string",
        "transaction_quantity": float,
        "price": float,
        "rate_units": "string",
        "standardized_quantity": float,
        "standardized_price": float,
        "total_transmission_charge": float,
        "total_transaction_charge": float,
    },
}

DATE_COLUMNS = {
    "contracts": [
        "contract_execution_date",
        "commencement_date_of_contract_term",
        "contract_termination_date",
        "actual_termination_date",
        "begin_date",
        "end_date",
    ],
    "transactions": [
        "transaction_begin_date",
        "transaction_end_date",
        "trade_date",
    ],
}
TRANSACT_AGG = {
    "by": [
        "seller_company_name",
        "customer_company_name",
        "ferc_tariff_reference",
        "contract_service_agreement",
        # "trade_date",
        "type_of_rate",
        "point_of_delivery_balancing_authority",
        "point_of_delivery_specific_location",
        "class_name",
        "term_name",
        "increment_name",
        "increment_peaking_name",
        "product_name",
        "rate_units",
    ],
    "agg": {
        # "transaction_unique_id": lambda x: ", ".join(set(x)),
        # "seller_company_name": set,
        # "customer_company_name": set,
        # "ferc_tariff_reference": set,
        # "contract_service_agreement": set,
        # "transaction_unique_identifier": lambda x: ", ".join(set(x)),
        # "transaction_begin_date": "first",
        # "transaction_end_date": "last",
        # "exchange_brokerage_service": "sum",
        # "type_of_rate": set,
        # "time_zone": set,
        # "point_of_delivery_balancing_authority": "first",
        # "point_of_delivery_specific_location": "first",
        # "class_name": set,
        # "term_name": set,
        # "increment_name": set,
        # "increment_peaking_name": set,
        # "product_name": set,
        # "transaction_quantity": "sum",
        # "price": "sum",
        # "rate_units": set,
        # "trade_date": "first",
        "standardized_quantity": "sum",
        # "standardized_price": "sum",
        "total_transmission_charge": "sum",
        "total_transaction_charge": "sum",
        "duration_hrs": "sum",
    },
}

WORKING_PARTITIONS = {"years": [2020], "quarters": ["Q1", "Q2", "Q3", "Q4"]}


class FercEqrPartition(NamedTuple):
    """Represents FercEqr partition identifying unique resource file."""

    year: int
    quarter: str


class ArgsPatch(NamedTuple):
    """Represents FercEqr partition identifying unique resource file."""

    years: list[int] = [2021]
    quarters: list[str] = ["Q1"]
    clobber: bool = True


def parse_command_line(argv):
    """Parse script command line arguments. See the -h option.

    Args:
        argv (list): command line arguments including caller file name.

    Returns:
        dict: A dictionary mapping command line arguments to their values.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        help="""Which years of FERC EQR data to process. Defaults to all years.""",
        default=WORKING_PARTITIONS["years"],
    )
    parser.add_argument(
        "-q",
        "--quarters",
        nargs="+",
        type=str.upper,
        help="""Which quarters to parocess. Defaults to all quarters.""",
        default=WORKING_PARTITIONS["quarters"],
    )
    parser.add_argument(
        "-c",
        "--clobber",
        action="store_true",
        default=False,
        help="Clobber existing EQR SQLite output.",
    )

    arguments = parser.parse_args(argv[1:])
    return arguments


def extract_seller(seller_zip: zipfile.ZipFile, partition) -> None:
    """
    Extract the tables and load them to a sqlite db for a seller.

    Args:
        seller_zip: A zipfile containing the tables for a single seller.
        partition: One quarter partition of EQR ferc data.
    """
    with seller_zip as seller:
        for table_type, table_name in FILE_END_STRS_TO_TABLE_NAMES.items():
            # find a file in seller_zip that matches the substring
            table_csv_path = list(
                filter(lambda x: x.endswith(table_type), seller.namelist())
            )
            assert len(table_csv_path) <= 1
            if table_csv_path:
                df = pd.read_csv(
                    io.BytesIO(seller_zip.read(table_csv_path[0])),
                    encoding="ISO-8859-1",
                    dtype=TABLE_DTYPES.get(table_name),
                    parse_dates=DATE_COLUMNS.get(table_name),
                )
                df["year"] = partition.year
                df["quarter"] = partition.quarter
                if table_name == "transactions":
                    # df = (
                    #     df.groupby(TRANSACT_AGG.get("by"))
                    #     .agg(TRANSACT_AGG.get("agg"))
                    #     .reset_index()
                    # )
                    fname = (
                        OUTPUTS_DIR
                        / "transactions"
                        / (
                            f"{partition.year}_{partition.quarter}_"
                            f"{table_csv_path[0].partition('_')[2].rpartition('_')[0]}.parquet"
                        )
                    )
                    if not df.empty and not fname.exists():
                        df.to_parquet(OUTPUTS_DIR / "transactions" / fname)

                else:

                    with engine.connect() as conn:
                        df.to_sql(table_name, conn, index=False, if_exists="append")


def extract_partition(partition: FercEqrPartition, multi=True, pool=4) -> None:
    """
    Extract a quarter of EQR data.

    Args:
        partition: One quarter partition of EQR ferc data.
    """
    quarter_zip_path = EQR_DATA_DIR / f"CSV_{partition.year}_{partition.quarter}.zip"
    if not quarter_zip_path.exists():
        raise FileNotFoundError(
            f"""
            Oops! It looks like that partition of data doesn't exist in {EQR_DATA_DIR}.
            Download the desired quarter from https://eqrreportviewer.ferc.gov/ to
            {EQR_DATA_DIR}.
            """
        )
    if multi:
        workers = []
        try:
            zipfile.ZipFile(quarter_zip_path, mode="r").extractall(
                quarter_zip_path.with_suffix("")
            )
            names = list(quarter_zip_path.with_suffix("").glob("*.ZIP"))
            k, m = divmod(len(names), 4)
            list_of_chunks = [names[i : i + k] for i in range(0, len(names), k)]
            for chunk in list_of_chunks:
                worker = mp.Process(target=_extract_helper_mp, args=(partition, chunk))
                workers.append(worker)
                worker.start()
        finally:
            for w in workers:
                w.join()
            shutil.rmtree(quarter_zip_path.with_suffix(""))
    else:
        with zipfile.ZipFile(quarter_zip_path, mode="r") as quarter_zip:
            _extract_helper(partition, quarter_zip, tqdm(quarter_zip.namelist()))


def _extract_helper_mp(partition, seller_paths):
    for seller_path in seller_paths:
        print(".", end="", flush=True)
        # seller_zip_bytes = io.BytesIO(quarter_zip.read(seller_path))
        seller_zip = zipfile.ZipFile(seller_path)
        extract_seller(seller_zip, partition)


def _extract_helper(partition, quarter_zip, seller_paths):
    for seller_path in seller_paths:
        seller_zip_bytes = io.BytesIO(quarter_zip.read(seller_path))
        seller_zip = zipfile.ZipFile(seller_zip_bytes)
        extract_seller(seller_zip, partition)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def aggregate_transactions(partitions: list[FercEqrPartition]):
    for partition in partitions:
        for file in tqdm(
            OUTPUTS_DIR.joinpath("transactions").glob(
                f"{partition.year}_{partition.quarter}_*.parquet"
            )
        ):
            df = pd.read_parquet(file).assign(
                duration_hrs=lambda x: (
                    x.transaction_end_date - x.transaction_begin_date
                ).dt.total_seconds()
                / 3600
            )
            df1 = (
                df.groupby(
                    TRANSACT_AGG.get("by")
                    + [
                        df.transaction_begin_date.dt.date,
                        df.transaction_end_date.dt.date,
                    ]
                )
                .agg(TRANSACT_AGG.get("agg"))
                .reset_index()
                .assign(
                    year=partition.year,
                    quarter=partition.quarter,
                    price=lambda x: (
                        x.total_transaction_charge - x.total_transmission_charge
                    )
                    / x.standardized_quantity,
                    product_name=lambda x: x.product_name.str.casefold(),
                )
            )

            with engine.connect() as conn:
                df1.to_sql("transactions", conn, index=False, if_exists="append")


def transaction_input_output():
    with engine.connect() as conn:
        tx = pd.read_sql_table("transactions", conn)
    tx = tx.assign(product_name=lambda x: x.product_name.str.casefold())
    inter = set(tx.customer_company_name).intersection(set(tx.seller_company_name))

    tx1 = (
        tx.query(
            "product_name == 'energy' & seller_company_name in @inter & customer_company_name in @inter"
        )
        .groupby(["seller_company_name", "customer_company_name"])[
            [
                "standardized_quantity",
                "total_transmission_charge",
                "total_transaction_charge",
                "duration_hrs",
            ]
        ]
        .sum()
        .reset_index()
    )
    x = tx1.pivot(
        index="seller_company_name",
        columns="customer_company_name",
        values="standardized_quantity",
    )
    # the inter thing doesn't actually guarantee it is square
    for i in set(x.columns) - set(x.index):
        x.loc[i, :] = np.nan
    for i in set(x.index) - set(x.columns):
        x.loc[:, i] = np.nan
    return x.sort_index(axis=1).sort_index(axis=0)


def main(args=None):
    """Load EQR contracts and identities to a sqlite database."""
    if args is None:
        args = parse_command_line(sys.argv)
    eqr_logger = logging.getLogger("pudl_rmi.process_eqr")
    log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
    coloredlogs.install(fmt=log_format, level="INFO", logger=eqr_logger)

    if args.clobber:
        EQR_DB_PATH.unlink(missing_ok=True)
    else:
        if EQR_DB_PATH.exists():
            raise SystemExit(
                "The FERC EQR DB already exists, and we don't want to clobber it.\n"
                f"Move {EQR_DB_PATH} aside or set clobber=True and try again."
            )

    (OUTPUTS_DIR / "transactions").mkdir(parents=True, exist_ok=True)

    partitions = [
        FercEqrPartition(year, quarter)
        for year in args.years
        for quarter in args.quarters
    ]

    for partition in tqdm(partitions):
        logger.info(f"Processing {partition}")
        extract_partition(partition)


if __name__ == "__main__":
    main()
