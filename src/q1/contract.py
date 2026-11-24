import copy


def with_contract(com, beta=None, alpha=None, cap_factor=None):
    # A copy of the community with other contract terms; the data files are never edited.
    #   beta        tariff discount (0.75 in the base case)
    #   alpha       penalty per kWh above the cap [DKK/kWh] (10 in the base case)
    #   cap_factor  cap energy factor f (1.3 in the base case): the cap is scaled by f / 1.3,
    #               and the individual caps of model I (kappa_i x cap) follow
    new = copy.copy(com)                      # the households are shared, only the contract is copied
    new.contract = copy.copy(com.contract)
    k = new.contract
    if beta is not None:
        k.tariff_discount_ratio = beta
    if alpha is not None:
        k.penalty_DKK_per_kWh = alpha
    if cap_factor is not None:
        scale = cap_factor / com.contract.capacity_limit_daily_energy_factor
        k.capacity_limit_kW = com.contract.capacity_limit_kW * scale
        k.capacity_limit_daily_energy_factor = cap_factor
        k.capacity_limit_daily_energy_kWh = float(k.capacity_limit_kW.sum())
        k.individual_capacity_limit_kW = {i: k.share[i] * k.capacity_limit_kW for i in com.member_ids}
    return new
