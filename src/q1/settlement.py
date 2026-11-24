import pandas as pd


def settlement(prices, hourly, members, members_R, members_I):
    # Settlement inside the community, household by household [DKK/day].
    #   prices                one internal price per hour [DKK/kWh]: lambda_t of model C, or the posted price of Q2
    #   hourly, members       the household tables of the community model (its C is only the PV cost there)
    #   members_R, members_I  the household tables of model R and model I
    # A household pays sum_t price_t * q_it, with q_it = p_im_it - p_ex_it its net use (q < 0: it supplies others).
    # Its full cost adds its own PV cost and disutility; gain > 0 means it is better off in the community.
    q = hourly["p_im"] - hourly["p_ex"]
    payment = pd.Series({i: (prices * q.loc[i]).sum() for i in members.index})
    s = pd.DataFrame({"archetype": members["archetype"], "payment": payment,
                      "cost": payment + members["C"] + members["D_L"],
                      "cost_R": members_R["total"], "cost_I": members_I["total"]})
    s["gain_vs_R"] = s["cost_R"] - s["cost"]
    s["gain_vs_I"] = s["cost_I"] - s["cost"]
    return s


def budget(prices, P_imp, P_exp, paid):
    # The community manager's budget [DKK/day]: collected from the households minus paid to the DSO.
    #   prices        one internal price per hour [DKK/kWh]
    #   P_imp, P_exp  import and export at the meter, per hour [kWh]
    #   paid          what the manager pays the DSO: energy + tariffs + penalty [DKK/day]
    collected = (prices * (P_imp - P_exp)).sum()
    return {"collected from members": collected, "paid to the DSO": paid, "surplus": collected - paid}
