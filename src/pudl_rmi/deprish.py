"""
Extract and transform steps for depreciation studies.

Catalyst has compiled depreciation studies for a project with the Rocky
Mountain Institue. These studies were compiled from Public Utility Commission
proceedings as well as the FERC Form 1 table.

how to run this module:
file_path_deprish = pathlib.Path().cwd().parent/'depreciation_rmi.xlsx'
sheet_name_deprish='Depreciation Studies Raw'
transformer = deprish.Transformer(
    deprish.Extractor(
        file_path=file_path_deprish,
        sheet_name=sheet_name_deprish
    ).execute())
deprish_df = transformer.execute()
deprish_asset_df = agg_to_idx(
    deprish_df,
    idx_cols=[x for x in IDX_COLS_DEPRISH if x not in ['ferc_acct', 'note']])
"""

import logging
from copy import deepcopy
import warnings
import pathlib

import pandas as pd
import numpy as np

import pudl
import pudl_rmi.make_plant_parts_eia as make_plant_parts_eia

logger = logging.getLogger(__name__)


INT_IDS = ['utility_id_ferc1', 'utility_id_pudl',
           'plant_id_pudl', 'report_year']

NA_VALUES = ["-", "—", "$-", ".", "_", "n/a", "N/A", "N/A $", "•", "*"]

IDX_COLS_DEPRISH = [
    'report_date',
    'plant_id_pudl',
    'plant_part_name',
    'ferc_acct',
    'note',
    'utility_id_pudl'
]

IDX_COLS_COMMON = [x for x in IDX_COLS_DEPRISH if x != 'plant_part_name']

# extract


class Extractor:
    """
    Extractor for turning excel based depreciation data into a dataframe.

    Note: this should be overhualed if/when we switch from storing the
    depreciation studies into a CSV. Also, if/when we integrate this into pudl,
    we need to think more seriously about where to store the excel sheet/CSV.
    Is it in pudl.package_data or do we store it through the datastore? If it
    felt stable would it be worthwhile to store via zendo?.. in which case we
    will want to use a datastore object to handle the path.
    """

    def __init__(self,
                 file_path,
                 sheet_name,
                 skiprows=0):
        """
        Initialize a for deprish.Extractor.

        Args:
            file_path (path-like)
            sheet_name (str, int): String used for excel sheet name or
                integer used for zero-indexed sheet location.
            skiprows (int): rows to skip in zero-indexed column location,
                default is 0.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.skiprows = skiprows

    def execute(self):
        """Turn excel-based depreciation data into a dataframe."""
        logger.info(f"Reading the depreciation data from {self.file_path}")
        return (
            pd.read_excel(
                self.file_path,
                skiprows=self.skiprows,
                sheet_name=self.sheet_name,
                dtype={i: pd.Int64Dtype() for i in INT_IDS},
                na_values=NA_VALUES
            )
            .dropna(axis='columns', how='all')
        )


class Transformer:
    """Transform class for cleaning depreciation study table."""

    def __init__(self, extract_df):
        """
        Initialize transform obect for cleaning depreciation study table.

        Args:
            extract_df (pandas.DataFrame): dataframe of extracted depreciation
                studies from ``Extractor.execute()``
        """
        # Note: should I pass in an instance of Extractor and make this call:
        # self.extract_df = extractor.execute()
        self.extract_df = extract_df

        self.tidy_df = None
        self.reshaped_df = None
        self.filled_df = None

    def execute(self, clobber=False):
        """
        Generate a transformed dataframe for the depreciation studies.

        Args:
            clobber (bool): if True and dataframe has already been generated,
                regenergate the datagframe.

        Returns:
            pandas.dataframe: depreciation study records that have been cleaned
            and nulls have been filled in.
        """
        self.tidy_df = self.early_tidy(clobber=clobber)
        self.reshaped_df = self.reshape(clobber=clobber)
        # value transform
        self.filled_df = self.fill_in(clobber=clobber)
        return self.filled_df

    def early_tidy(self, clobber=False):
        """Early transform type assignments and column assignments."""
        if clobber or self.tidy_df is None:
            # read in the depreciation sheet, assign types when required
            # we need the dtypes assigned early in this process because the
            # next steps involve splitting and filling in the null columns.
            self.tidy_df = (
                self.extract_df
                .pipe(self._convert_rate_cols)
                .pipe(pudl.helpers.convert_cols_dtypes,
                      'depreciation', name='depreciation')
                .assign(report_year=lambda x: x.report_date.dt.year)
                .pipe(pudl.helpers.simplify_strings, ['plant_part_name'])
                .pipe(add_ferc_acct_name)
                .pipe(assign_line_id)
            )
        return self.tidy_df

    def reshape(self, clobber=False):
        """
        Structural transformations.

        Right now, this implements ``split_allocate_common()`` which grabs the
        common records out of the main df and associates relevant dat columns
        with the related non-common records. In the end, we have a table that
        has no more common rows and a few extra columns that have neatly
        associated the common rows' data. We may need different types of
        reshaping later, so this method is here to accumulate reshaping
        methods.
        """
        if clobber or self.reshaped_df is None:
            self.reshaped_df = self.split_allocate_common()
        return self.reshaped_df

    def fill_in(self, clobber=False):
        """
        Clean % columns and fill in missing values.

        Args:
            clobber (bool): if True and dataframe has already been generated,
                regenergate the datagframe.

        Returns:
            pandas.DataFrame: depreciation study records that have been cleaned
            and nulls have been filled in.
        """
        if clobber or self.filled_df is None:
            filled_df = deepcopy(self.reshape())
            # convert % columns - which originally are a combination of whole
            # numbers of decimals (e.g. 88.2% would either be represented as
            # 88.2 or .882). Some % columns have boolean columns (ending in
            # type_pct) that we fleshed out to know wether the values were
            # reported as numbers or %s. There is one column that was easy to
            # clean by checking whether or not the value is greater than 1.
            filled_df.loc[~filled_df['net_salvage_rate_type_pct'],
                          'net_salvage_rate'] = (
                filled_df.loc[~filled_df['net_salvage_rate_type_pct'],
                              'net_salvage_rate'] * 100
            )
            filled_df.loc[filled_df['depreciation_annual_rate_type_pct'],
                          'depreciation_annual_rate'] = (
                filled_df.loc[filled_df['depreciation_annual_rate_type_pct'],
                              'depreciation_annual_rate'] / 100
            )
            filled_df.loc[abs(filled_df.reserve_rate) >= 1,
                          'reserve_rate'] = filled_df.loc[
                abs(filled_df.reserve_rate) >= 1, 'reserve_rate'] / 100
            logger.info(
                f"# of reserve_rate over 1 (100%): "
                f"{len(filled_df.loc[abs(filled_df.reserve_rate) >= 1])} "
                "Higher #s here may indicate an issue with the original data "
                "or the fill_in method"
            )
            # get rid of the bool columns we used to clean % columns
            filled_df = filled_df.drop(
                columns=filled_df.filter(like='num'))

            filled_df['net_salvage_rate'] = (
                - filled_df['net_salvage_rate'].abs()
            )
            filled_df['net_salvage'] = - filled_df['net_salvage'].abs()

            # then we need to do the actuall filling in
            def _fill_in_assign(filled_df):
                return filled_df.assign(
                    net_salvage_rate=lambda x:
                        # first clean % v num, then net_salvage/book_value
                        x.net_salvage_rate.fillna(
                            x.net_salvage / x.book_reserve),
                    net_salvage=lambda x:
                        x.net_salvage.fillna(
                            x.net_salvage_rate * x.book_reserve),
                    book_reserve=lambda x: x.book_reserve.fillna(
                        x.plant_balance_w_common -
                        (x.depreciation_annual_epxns * x.remaining_life_avg)),
                    unaccrued_balance=lambda x:
                        x.unaccrued_balance.fillna(
                            x.plant_balance_w_common - x.book_reserve
                            - x.net_salvage),  # ??
                    reserve_rate=lambda x: x.book_reserve /
                    x.plant_balance_w_common,
                )
            # we want to do this filling in twice because the order matters.
            filled_df = _fill_in_assign(filled_df).pipe(_fill_in_assign)
            self.filled_df = filled_df

        return self.filled_df

    def _convert_rate_cols(self, tidy_df):
        """Convert percent columns to numeric."""
        to_num_cols = ['net_salvage_rate',
                       'reserve_rate',
                       'depreciation_annual_rate']
        for col in to_num_cols:
            tidy_df[col] = pd.to_numeric(tidy_df[col])
        return tidy_df

    def split_merge_common_records(self, split_col):
        """
        Split apart common records and merge back specific columns.

        Args:
            split_col (string): name of column

        Returns:
            pandas.DataFrame: the depreciation data with plant_balance_common,
                a count of instances of the common records and main records.

        """
        common_pb = self.get_common_plant_bal(split_col=split_col)

        deprish_w_c = (
            pd.merge(
                self.early_tidy(),
                common_pb,
                left_on=['line_id', 'ferc_acct'],
                right_on=['line_id_main', 'ferc_acct'],
                how='left',
                suffixes=('', '_common')
            )
            .drop(columns=['line_id_main'])
        )
        # at this stage we have merged in the plant_balance of the common
        # records with their associated main depreciation records, so we can
        # remove the common records from the main depreciation table.
        deprish_w_c = deprish_w_c.loc[
            ~deprish_w_c.line_id.isin(common_pb.line_id_common.unique())]

        return deprish_w_c

    def get_common_plant_bal(self, split_col='plant_balance'):
        """Get."""
        common_assn = get_common_assn()
        self.common_len = len(common_assn.line_id_common)
        # merge back in the ferc acct #
        common_assn_acct = (
            pd.merge(
                common_assn,
                self.early_tidy()[['line_id', 'ferc_acct']],
                left_on=['line_id_common'],
                right_on=['line_id'],
            )
            .drop(columns=['line_id'])
        )

        common_pb = (
            pd.merge(
                common_assn_acct,
                self.early_tidy()[['line_id', 'ferc_acct', split_col]],
                left_on=['line_id_common', 'ferc_acct'],
                right_on=['line_id', 'ferc_acct'],
                how='left'
            )
            .drop(columns=['line_id'])
            .drop_duplicates()
            .dropna(subset=['line_id_common', 'line_id_main', split_col])
            .pipe(self._count_common_assn)
        )
        return common_pb

    def _count_common_assn(self, common_pb):
        common_pb_w_counts = (
            common_pb
            .merge(
                (
                    common_pb.assign(count_common=1)
                    .groupby(['line_id_common', 'ferc_acct'])
                    [['count_common']].sum().reset_index()
                ),
                on=['line_id_common', 'ferc_acct']
            )
            .merge(
                (
                    common_pb.assign(count_main=1)
                    .groupby(['line_id_main', 'ferc_acct'])
                    [['count_main']].sum().reset_index()
                ),
                on=['line_id_main', 'ferc_acct']
            )
        )
        return common_pb_w_counts

    def split_allocate_common(self,
                              split_col='plant_balance',
                              common_suffix='_common'):
        """
        Split and allocate the common plant depreciation lines.

        The depreciations studies have common plant records sprinkled
        throughout, which represent the shared infrastructure (read capital in
        this context) of a plant with multiple units. Because we care about the
        sub-units of a plant, we don't actually care about the individual
        common records. We want to distribute the undepreciated plant balances
        associated with "common" records that pertain to no generation unit in
        particular, across all generation units, in proportion to each unit's
        own remaining plant balance.

        Args:
            split_col (string): column name of common records to split and
                allocate. Column must contain numeric data. Default
                'plant_balance'.
            common_suffix (string): suffix to use for the common columns when
                they are merged into the other plant-part records.
        """
        # the new  data col we are trying to generate
        new_data_col = f'{split_col}_w{common_suffix}'

        deprish_w_c = self.split_merge_common_records(split_col=split_col)

        simple_case_df = self.calc_common_portion_simple(
            deprish_w_c, split_col, common_suffix, new_data_col)
        edge_case_df = self.calc_common_portion_with_no_part_balance(
            deprish_w_c, split_col, common_suffix, new_data_col)

        deprish_w_common_allocated = pd.concat([simple_case_df, edge_case_df])

        # finally, calcuate the new column w/ the % of the total group. if
        # there is no common data, fill in this new data column with the og col
        deprish_w_common_allocated[new_data_col] = (
            deprish_w_common_allocated[f"{split_col}_common_portion"].fillna(0)
            + deprish_w_common_allocated[split_col].fillna(0))

        if len(deprish_w_common_allocated) != len(deprish_w_c):
            raise AssertionError(
                "smh.. the number of alloacted records "
                f"({len(deprish_w_common_allocated)}) don't match the "
                f"original records ({len(deprish_w_c)})... "
                "so something went wrong here."
            )
        _ = self._check_common_allocation(
            deprish_w_common_allocated, split_col, new_data_col, common_suffix)

        return deprish_w_common_allocated

    def calc_common_portion_simple(self,
                                   deprish_w_c,
                                   split_col,
                                   common_suffix,
                                   new_data_col):
        """
        Generate the portion of the common plant based on the split_col.

        Most of the deprecation records have data in our default ``split_col``
        (which is ``plant_balance``). For these records, calculating the
        portion of the common records to allocate to each plant-part is simple.
        This method calculates the portion of the common plant balance that
        should be allocated to each plant-part/ferc_acct records based on the
        ratio of each records' plant balance compared to the total
        plant/ferc_acct plant balance.
        """
        # exclude the nulls and the 0's
        simple_case_df = deprish_w_c[
            (deprish_w_c[split_col].notnull()) & (deprish_w_c[split_col] != 0)
        ]
        logger.info(
            f"We are calculating the common portion for {len(simple_case_df)} "
            f"records w/ {split_col}")

        # we want to know the sum of the potential split_cols for each ferc1
        # option
        gb_df = (
            simple_case_df
            .groupby(by=IDX_COLS_COMMON, dropna=False)
            [[split_col]].sum(min_count=1).reset_index()
        )

        df_w_tots = (
            pd.merge(
                simple_case_df,
                gb_df,
                on=IDX_COLS_COMMON,
                how='left',
                suffixes=("", "_sum"))
        )

        df_w_tots[f"{split_col}_ratio"] = (
            df_w_tots[split_col] / df_w_tots[f"{split_col}_sum"]
        )

        # the default way to calculate each plant sub-part's common plant
        # portion is to multiply the ratio (calculated above) with the total
        # common plant balance for the plant/ferc_acct group.
        df_w_tots[f"{split_col}_common_portion"] = (
            df_w_tots[f'{split_col}{common_suffix}']
            * df_w_tots[f"{split_col}_ratio"])

        return df_w_tots

    def calc_common_portion_with_no_part_balance(self,
                                                 deprish_w_c,
                                                 split_col,
                                                 common_suffix,
                                                 new_data_col):
        """
        Calculate portion of common when ``split_col`` is null.

        There are a handfull of records where there is ``split_col`` values
        from the common records, but the ``split_col`` for that plant sub-part
        is null. In these cases, we still want to check if we need to assocaite
        a portion of the common ``split_col`` should be broken up based on the
        number of other records that the common value is assocaitd with
        (within the group of the ``IDX_COLS_COMMON``). We check to see if
        there are other plant sub-parts in the common plant grouping that have
        non-zero/non-null ``split_col`` - if they do then we don't assign the
        common portion to these records because their record relatives will be
        assigned the full common porportion in the
        ``calc_common_portion_simple()``.
        """
        # there are a handfull of records which have no plant balances
        # but do have common plant_balances.
        edge_case_df = deprish_w_c[
            (deprish_w_c[split_col].isnull()) | (deprish_w_c[split_col] == 0)
        ]

        logger.info(
            f"We are calculating the common portion for {len(edge_case_df)} "
            f"records w/o {split_col}")

        # for future manipliations, we want a count of the number of records
        # within each group and have a bool column that lets us know whether or
        # not any of the records in a group have a plant balance
        edge_case_count = (
            deprish_w_c
            .assign(
                plant_bal_count=1,
                plant_bal_any=np.where(
                    deprish_w_c.plant_balance > 0,
                    True, False)
            )
            .groupby(by=IDX_COLS_COMMON, dropna=False)
            .agg({'plant_bal_count': 'count',
                  'plant_bal_any': 'any'})
        )
        edge_case_df = pd.merge(
            edge_case_df,
            edge_case_count.reset_index(),
            on=IDX_COLS_COMMON,
            how='left'
        )
        # if there is no other plant records with plant balances in the same
        # plant/ferc_acct group (denoted by the plant_bal_any column), we split
        # the plant balance evenly amoung the records using plant_bal_count.
        # if there are other plant sub part records with plant balances, the
        # common plant balance will already be distributed amoung those records
        edge_case_df[f"{split_col}_common_portion"] = np.where(
            ~edge_case_df['plant_bal_any'],
            (edge_case_df[f'{split_col}{common_suffix}'] /
             edge_case_df['plant_bal_count']),
            np.nan
        )

        return edge_case_df

    def _check_common_allocation(self,
                                 df_w_tots,
                                 split_col,
                                 new_data_col,
                                 common_suffix):
        """Check to see if the common plant allocation was effective."""
        calc_check = (
            df_w_tots
            .groupby(by=IDX_COLS_DEPRISH, dropna=False)
            [[f"{split_col}_ratio", f"{split_col}_common_portion"]]
            .sum(min_count=1)
            .add_suffix("_check")
            .reset_index()
        )
        df_w_tots = pd.merge(
            df_w_tots, calc_check, on=IDX_COLS_DEPRISH, how='outer'
        )

        df_w_tots[f"{split_col}_common_portion_check"] = np.where(
            (df_w_tots.plant_balance.isnull() &
             df_w_tots.plant_balance_common.notnull()),
            df_w_tots[f"{split_col}_common_portion"] *
            df_w_tots["plant_bal_count"],
            df_w_tots[f"{split_col}_common_portion_check"]
        )

        # sum up all of the slices of the plant balance column.. these will be
        # used in the logs/asserts below
        plant_balance_og = (
            self.tidy_df[self.tidy_df.plant_part_name.notnull()]
            [split_col].sum()
        )
        plant_balance = df_w_tots[split_col].sum()
        plant_balance_w_common = df_w_tots[df_w_tots.plant_part_name.notnull(
        )][new_data_col].sum()
        plant_balance_c = (
            df_w_tots.drop_duplicates(
                subset=[c for c in IDX_COLS_DEPRISH if c != 'plant_part_name'],
                keep='first')
            [f"{split_col}{common_suffix}"].sum())

        logger.info(
            f"The resulting {split_col} allocated is "
            f"{plant_balance_w_common / plant_balance_og:.02%} of the original"
        )
        if plant_balance_w_common / plant_balance_og < .99:
            warnings.warn(
                f"ahhh the {split_col} allocation is off. The resulting "
                f"{split_col} is "
                f"{plant_balance_w_common/plant_balance_og:.02%} of the "
                f"original. og {plant_balance_og:.3} vs new: "
                f"{plant_balance_w_common:.3}"
            )

        if (plant_balance + plant_balance_c) / plant_balance_og < .99:
            warnings.warn(
                "well something went wrong here. even before proportionally "
                "assigning the common plant balance, the plant balance + "
                "common doesn't add up."
            )

        if len(df_w_tots) + self.common_len != len(self.early_tidy()):
            warnings.warn(
                'ahhh we have a problem here with the number of records being '
                'generated here'
            )

        bad_ratio_check = (df_w_tots[~df_w_tots['plant_balance_ratio_check']
                                     .round(0).isin([1, 0])])
        if not bad_ratio_check.empty:
            warnings.warn(
                f"why would you do this?!?! there are {len(bad_ratio_check)} "
                f"records that are not passing our {split_col} check. "
                "The common records are being split and assigned incorrectly. "
            )
        # check for records w/ associated common plant balance
        no_common = df_w_tots[
            (df_w_tots.plant_balance_common.isnull()
             & (df_w_tots.plant_balance.notnull()))
            & (df_w_tots.plant_balance != df_w_tots.plant_balance_w_common)
        ]
        if not no_common.empty:
            warnings.warn(
                f"Ack! We have {len(no_common)} records that have no common "
                f"{split_col} but the og {split_col} is different than "
                f"the {new_data_col}"
            )
        return df_w_tots


def agg_to_idx(deprish_df, idx_cols):
    """
    Aggregate the depreciation data to the asset level.

    The depreciation data is reported at the plant-part and ferc_acct level.

    Args:
        deprish_df (pandas.DataFrame): table of depreciation data at the
            plant_part_name/ferc_acct level. Result of
            `Transformer().execute()`.

    Returns:
        pandas.DataFrame: table of depreciation data scaled down to the asset
        level. This functionally removes the granularity of the FERC account #.

    """
    # prep for the groupby:
    # we have to break out the columns that need to be summed and the columns
    # which needs to be run through a weighted average aggregation for two
    # reasons. we need to insert the min_count=1 for the summed columns so we
    # don't end up with a bunch of 0's when it should be nulls. and because
    # there isn't a built in weighted average gb.agg function, so we need to
    # run it through our own.

    # sum agg section
    # enumerate sum cols
    sum_cols = [
        'plant_balance', 'book_reserve',
        'unaccrued_balance', 'net_salvage', 'depreciation_annual_epxns', ]
    if 'plant_balance_w_common' in deprish_df.columns:
        sum_cols.append(['plant_balance_w_common'])
    # aggregate..
    deprish_asset = deprish_df.groupby(by=idx_cols, dropna=False)[
        sum_cols].sum(min_count=1)

    # weighted average agg section
    # enumerate wtavg cols
    avg_cols = ['service_life_avg', 'remaining_life_avg'] + \
        [x for x in deprish_df.columns
         if '_rate' in x and 'rate_type_pct' not in x]
    # prep dict with col to average (key) and col to weight on (value)
    # in this case we always want to weight based on unaccrued_balance
    wtavg_cols = {}
    for col in avg_cols:
        wtavg_cols[col] = 'unaccrued_balance'
    # aggregate..
    for data_col, weight_col in wtavg_cols.items():
        deprish_asset = (
            deprish_asset.merge(
                make_plant_parts_eia.weighted_average(
                    deprish_df,
                    data_col=data_col,
                    weight_col=weight_col,
                    by_col=idx_cols),
                # .rename(columns={data_col: f"{data_col}_wt"})
                how='outer', on=idx_cols))

    if 'plant_balance_w_common' in deprish_df.columns:
        deprish_asset = deprish_asset.assign(
            remaining_life_avg=lambda x:
                x.unaccrued_balance / x.depreciation_annual_epxns,
            plant_balance_w_common_check=lambda x:
                x.book_reserve + x.unaccrued_balance,
            plant_balance_diff_check=lambda x:
                x.plant_balance_w_common_check / x.plant_balance_w_common,
        )

    return deprish_asset


def fill_in_tech_type(gens):
    """
    Fill in the generators' tech type based on energy source and prime mover.

    Args:
        gens (pandas.DataFrame): generators_eia860 table
    """
    # back fill the technology type
    idx_es_pm_tech = [
        'energy_source_code_1', 'prime_mover_code', 'technology_description'
    ]
    es_pm = ['energy_source_code_1', 'prime_mover_code']
    gens_f_pm_t = (
        gens.groupby(idx_es_pm_tech)
        [['plant_id_eia']].count().add_suffix('_count').reset_index()
    )

    logger.info(
        f"{len(gens_f_pm_t[gens_f_pm_t.duplicated(subset=es_pm)])} "
        "duplicate tech type mappings")
    tech_type_map = (
        gens_f_pm_t.sort_values('plant_id_eia_count', ascending=False)
        .drop_duplicates(subset=es_pm)
        .drop(columns=['plant_id_eia_count']))

    gens = (
        pd.merge(
            gens,
            tech_type_map,
            on=es_pm,
            how='left',
            suffixes=("", "_map"),
            validate="m:1"
        )
        .assign(technology_description=lambda x:
                x.technology_description.fillna(x.technology_description_map))
        .drop(columns=['technology_description_map'])
    )

    no_tech_type = gens[gens.technology_description.isnull()]
    logger.info(
        f"{len(no_tech_type)/len(gens):.01%} of generators don't map to tech "
        "types"
    )
    return gens

######################
# Line ID Generation #
######################


def get_ferc_acct_type_map(file_path):
    """Grab the mapping of the FERC Account numbers to names."""
    ferc_acct_map = (
        pd.read_csv(file_path, dtype={'ferc_acct': pd.StringDtype()})
    )
    return ferc_acct_map


def add_ferc_acct_name(tidy_df):
    """Add the FERC Account name into the tidied deprecation table."""
    file_path_ferc_acct_names = (
        pathlib.Path().cwd().parent / 'inputs' / 'ferc_acct_names.csv')
    ferc_acct_names = get_ferc_acct_type_map(
        file_path_ferc_acct_names)

    # break out the float-y decimals in the ferc acct col into a sub column
    tidy_df[['ferc_acct', 'ferc_acct_sub']] = (
        tidy_df.ferc_acct.str.split('.', expand=True))
    tidy_df = (
        pd.merge(
            tidy_df,
            ferc_acct_names[['ferc_acct', 'ferc_acct_name']],
            on=['ferc_acct'],
            how='left',
            validate='m:1'
        )
    )
    return tidy_df


def assign_line_id(df):
    """Make a composite id column."""
    df = df.assign(
        line_id=lambda x:
        x.report_date.dt.year.astype(pd.Int64Dtype()).map(str) + "_" +
        x.plant_id_pudl.map(str) + "_" +
        x.plant_part_name.map(str).str.lower() + "_" +
        x.ferc_acct_name.fillna("").str.lower() + "_" +
        x.note.fillna("") + "_" +
        x.utility_id_pudl.map(str) + "_" +
        x.data_source.fillna("")
    )
    return df

#################################
# Common Association & Labeling #
#################################


def get_common_assn():
    """Get stored common plant assocations."""
    path_common_assn = pathlib.Path().cwd().parent / 'outputs/common_assn.csv'
    common_assn = pd.read_csv(path_common_assn)
    return common_assn


def make_common_assn_for_labeling(common_assn, pudl_out, transformer):
    """Make."""
    common_assn_wide = transform_common_assn_for_labeling(common_assn)
    plants_pudl = get_plant_pudl_info(pudl_out)
    common_labeling = (
        pd.merge(
            agg_to_idx(transformer.early_tidy(),
                       idx_cols=['line_id', 'plant_id_pudl', 'report_date']),
            plants_pudl,
            on=['plant_id_pudl', 'report_date'],
            how='left',
            validate='m:1',
            suffixes=('', '_eia')
        )
        .set_index(['line_id'])
        .assign(common=pd.NA)
        .merge(
            common_assn_wide,
            right_index=True,
            left_index=True,
            how='outer',
            indicator=True
        )
        .assign(
            common=lambda x: np.where(
                x._merge == 'both', True, False),
            ignore=pd.NA,
            plant_id_pudl_off=pd.NA
        )
        .drop(columns=['_merge'])
    )

    common_labeling.index.name = 'line_id'
    return common_labeling


def get_plant_pudl_info(pudl_out):
    """Grab info about plants, aggregated to plant_id_pudl/report_date."""
    plants_pudl = (
        pudl_out.gens_eia860()
        .assign(count='place_holder')
        .sort_values(['plant_name_eia', 'state'])
        .groupby(['plant_id_pudl', 'report_date'], as_index=False)
        # Must use .join because x.unique() arrays are not hashable
        .agg(
            {'plant_id_eia':
             lambda x: '; '.join([str(x) for x in x.unique() if x]),
             'generator_id': lambda x: '; '.join(x.unique()),
             'count': lambda x: x.count(),
             'capacity_mw': lambda x: x.sum(min_count=1),
             'plant_name_eia': lambda x: x.iloc[0],
             'state': lambda x: x.iloc[0],
             'operational_status':
             lambda x: '; '.join([x for x in x.unique() if x]),
             'fuel_type_code_pudl':
             lambda x: '; '.join([x for x in x.unique() if x]),
             })
        .astype({'plant_id_pudl': 'Int64'})
    )
    return plants_pudl


def transform_common_assn_for_labeling(common_assn):
    """
    Convert the skinny common assn into a wide version for mannual labeling.

    Args:
        common_assn (pandas.DataFrame): skinny association table with two
            columns: line_id_common & line_id_main.
    Return:
        pandas.DataFrame:

    """
    common_assn = common_assn.sort_values(['line_id_common'])
    to_array = (
        common_assn.groupby('line_id_common')
        ['line_id_main'].unique().tolist()
    )
    new_df = (
        pd.DataFrame(to_array)
        .set_index(np.array(common_assn['line_id_common'].unique()))
    )
    common_assn_wide = new_df.rename(
        columns={n: 'line_id_main_' + str(n + 1) for n in new_df.columns})
    return common_assn_wide


##############################
# Default Common Association #
##############################


def make_default_common_assn(file_path_deprish):
    """
    Generate default common associations.

    Grab the compiled depreciation data, get the default common records which
    have 'common' in their name, make associations with the non-common records
    based on IDX_COLS_COMMON.
    """
    transformer = Transformer(
        extract_df=Extractor(file_path=file_path_deprish,
                             sheet_name=0).execute()
    )
    # assume common plant records based on the plant_part_name
    deprish_df = (
        transformer.early_tidy()
        .assign(
            common=lambda x:
            x.plant_part_name.fillna(
                'fake name so the col wont be null')
            .str.contains('common|comm')
        )
    )

    # if there is no plant_id_pudl, there will be no plant for the common
    # record to be allocated across, so for now we need to assume these
    # records are not common
    deprish_c_df = deprish_df.loc[
        deprish_df.common & deprish_df.plant_id_pudl.notnull()
    ]
    deprish_df = deprish_df.loc[
        ~deprish_df.common | deprish_df.plant_id_pudl.isnull()]

    common_assn = (
        pd.merge(
            deprish_c_df,
            deprish_df,
            how='left',
            on=IDX_COLS_COMMON,
            suffixes=("_common", "_main")
        )
        .filter(like='line')
        .sort_values(['line_id_common'])
        .drop_duplicates()
    )
    return common_assn