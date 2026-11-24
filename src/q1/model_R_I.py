""" Models R and I - every household alone behind its own meter, each solving its own FlexibleConsumer problem. """
import pandas as pd

from src.q1.consumer import FlexibleConsumer, collect
from src.data_loader import load_community


def solve_members(com, contract=None):
    """Solve every household alone: model R if contract is None, model I if contract = com.contract.
    Returns (members, hourly), the two tables of consumer.collect()."""
    consumers = [FlexibleConsumer(d, contract) for d in com.members]
    for c in consumers:
        c.solve()
    return collect(consumers)


def feeder(hourly, cap):
    """What the DSO meters under R and I, per hour: the import P_imp = sum_i p_im_it [kW], the cap, and the
    exceedance max(0, P_imp - cap) [kW]."""
    P_imp = hourly["p_im"].groupby("t").sum()
    return pd.DataFrame({"P_imp": P_imp, "cap": cap, "exceedance": (P_imp - cap).clip(lower=0)})


if __name__ == "__main__":
    com = load_community("base", day="d1")
    members, hourly = solve_members(com)
    f = feeder(hourly, com.contract.capacity_limit_kW)
    ref = com.contract.reference_model_R

    print("Check of model R against dso_contract_params.json (base case, d1)")
    print(f"  daily import [kWh]   {f['P_imp'].sum():8.2f}   file {ref['daily_import_kWh']:8.2f}")
    print(f"  peak import [kW]     {f['P_imp'].max():8.2f}   file {ref['peak_import_kW']:8.2f}")
    print(f"  cost [DKK]           {members['total'].sum():8.2f}   file {ref['cost_DKK']:8.2f}")
    print(f"  exceedance [kWh]     {f['exceedance'].sum():8.2f}   file {ref['exceedance_kWh']:8.2f}")
    net_import = (hourly["p_im"] - hourly["p_ex"]).groupby("member").sum()
    file_net = pd.Series(com.contract.net_import_in_R_kWh)
    print(f"  largest difference in a household's net import: {(net_import - file_net).abs().max():.2f} kWh")
