"""One household (member) of the community: the Assignment 1 consumer, adapted to Assignment 2. A linear program (LP).

Built from three Assignment 1 solution classes (same notation, same constraint form, same equation numbers):
    linear disutility           consumer_q2_linear.py   (2.1)-(2.3)
    minimum daily energy        consumer_q3.py          (3.2)
    battery                     consumer_q3_battery.py  (3.6)-(3.11)
The load bounds (1.4) are replaced by the hourly bounds of Assignment 2 (A2.1).

    min  sum_t  pi_im_t p_im_t - pi_ex_t p_ex_t + c_PV p_PV_t + c_L (d_pos_t + d_neg_t)
    s.t. l_t + p_ex_t + p_ch_t - p_im_t - p_PV_t - p_dis_t = 0                                    (3.6)
         p_PV_t - PV_bar_t <= 0,  -p_PV_t <= 0                                                    (1.3)
         l_t - L_up_t <= 0,  L_lo_t - l_t <= 0,  L_lo_t = max(0, l_ref_t - 1), L_up_t = min(3, l_ref_t + 1)   (A2.1)
         -p_im_t <= 0,  -p_ex_t <= 0                                                              (1.5)
         l_t - l_ref_t - d_pos_t + d_neg_t = 0,  -d_pos_t <= 0,  -d_neg_t <= 0                    (2.2)-(2.3)
         E_min - sum_t l_t <= 0                                                                   (3.2)
         battery: state of charge (3.7)-(3.8), limits (3.9)-(3.10), end of day S_init - s_24 <= 0  (3.11)
         (a household without a battery gets a battery of size 0, so p_ch = p_dis = s = 0)

The same class is used in the three models of Question 1:
    FlexibleConsumer(d)                          model R: alone, full tariffs pi_im = pi_DA + tau_imp, pi_ex = pi_DA - tau_exp
    FlexibleConsumer(d, contract=com.contract)   model I: alone, tariffs x (1 - beta), and its own cap with a penalty:
                                                     + alpha e_t in the objective
                                                     p_im_t - cap_i,t - e_t <= 0,  -e_t <= 0           (A2.2)-(A2.3)
    FlexibleConsumer(d, model=m)                 model C: added to the community's model m (src/q1/model_C.py); it pays no
                                                     grid cost itself (the manager pays at the meter), only its PV cost

Conventions as in Assignment 1: every variable unbounded, every bound written as its own constraint.
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


class FlexibleConsumer:

    def __init__(self, data, contract=None, model=None):
        self.data = data
        self.contract = contract                      # None: model R; com.contract: model I
        self.in_community = model is not None         # True: model C
        self.T = range(1, data.n_hours + 1)           # hours t = 1, ..., 24
        if self.in_community:
            self.m = model                            # the community's model (C)
        else:
            self.m = gp.Model(f"consumer_{data.consumer_id}")
            self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T
        i = d.consumer_id                             # used in the names, so 16 households can share one model (C)

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        if self.contract is None:
            tariff_factor = 1.0                                       # R: full tariffs
        else:
            tariff_factor = 1.0 - self.contract.tariff_discount_ratio  # I: (1 - beta) = 0.25
        pi_DA = {t: d.energy_price[t - 1] for t in T}
        pi_im = {t: pi_DA[t] + tariff_factor * d.import_tariff for t in T}
        pi_ex = {t: pi_DA[t] - tariff_factor * d.export_tariff for t in T}
        PV_bar = {t: d.pv_available[t - 1] for t in T}                # 0 in every hour for households without PV
        l_ref = {t: d.reference_load[t - 1] for t in T}
        L_lo = {t: d.load_lower_kWh[t - 1] for t in T}                # max(0, l_ref - 1)
        L_up = {t: d.load_upper_kWh[t - 1] for t in T}                # min(3, l_ref + 1)
        c_PV, c_L, E_min = d.pv_marginal_cost, d.linear_disutility, d.min_daily_energy_kWh
        if d.has_battery:
            S_bar, S_init = d.battery_capacity_kWh, d.battery_initial_soc_kWh
            P_ch_bar, P_dis_bar = d.battery_max_charge_kW, d.battery_max_discharge_kW
            eta_ch, eta_dis = d.battery_charging_efficiency, d.battery_discharging_efficiency
        else:                                                         # no battery = a battery of size 0
            S_bar, S_init, P_ch_bar, P_dis_bar = 0.0, 0.0, 0.0, 0.0
            eta_ch, eta_dis = 1.0, 1.0

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        p_im = self.p_im = m.addVars(T, lb=-GRB.INFINITY, name=f"p_im_{i}")
        p_ex = self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name=f"p_ex_{i}")
        p_PV = self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name=f"p_PV_{i}")
        l = self.l = m.addVars(T, lb=-GRB.INFINITY, name=f"l_{i}")
        d_pos = self.d_pos = m.addVars(T, lb=-GRB.INFINITY, name=f"d_pos_{i}")
        d_neg = self.d_neg = m.addVars(T, lb=-GRB.INFINITY, name=f"d_neg_{i}")
        p_ch = self.p_ch = m.addVars(T, lb=-GRB.INFINITY, name=f"p_ch_{i}")
        p_dis = self.p_dis = m.addVars(T, lb=-GRB.INFINITY, name=f"p_dis_{i}")
        s = self.s = m.addVars(T, lb=-GRB.INFINITY, name=f"s_{i}")

        # Cost of the household: procurement cost C, penalty P (model I only), disutility D_L
        if self.in_community:       # C: the manager pays the grid at the meter, the household only its PV
            self.C = gp.quicksum(c_PV * p_PV[t] for t in T)
        else:                       # R, I: the household pays its own meter
            self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.D = gp.quicksum(c_L * (d_pos[t] + d_neg[t]) for t in T)
        if self.contract is None:
            self.P = gp.LinExpr()   # no contract, no penalty
        else:                       # I: own cap kappa_i x Pbar_t, penalty alpha on the exceedance e_t
            cap_i = {t: self.contract.individual_capacity_limit_kW[i][t - 1] for t in T}
            alpha = self.contract.penalty_DKK_per_kWh
            e = self.e = m.addVars(T, lb=-GRB.INFINITY, name=f"e_{i}")
            m.addConstrs((p_im[t] - cap_i[t] - e[t] <= 0 for t in T), name=f"cap_{i}")             # (A2.2)
            m.addConstrs((-e[t] <= 0 for t in T), name=f"e_lo_{i}")                                # (A2.3)
            self.P = gp.quicksum(alpha * e[t] for t in T)
        if not self.in_community:   # in C, the community model sets the objective
            m.setObjective(self.C + self.P + self.D, GRB.MINIMIZE)

        # Constraints, written as h(x) = 0 or g(x) <= 0
        m.addConstrs((l[t] + p_ex[t] + p_ch[t] - p_im[t] - p_PV[t] - p_dis[t] == 0 for t in T), name=f"bal_{i}")  # (3.6)
        m.addConstrs((p_PV[t] - PV_bar[t] <= 0 for t in T), name=f"PV_up_{i}")                         # (1.3)
        m.addConstrs((-p_PV[t] <= 0 for t in T), name=f"PV_lo_{i}")                                    # (1.3)
        m.addConstrs((l[t] - L_up[t] <= 0 for t in T), name=f"l_up_{i}")                              # (A2.1)
        m.addConstrs((L_lo[t] - l[t] <= 0 for t in T), name=f"l_lo_{i}")                              # (A2.1)
        m.addConstrs((-p_im[t] <= 0 for t in T), name=f"im_lo_{i}")                                    # (1.5)
        m.addConstrs((-p_ex[t] <= 0 for t in T), name=f"ex_lo_{i}")                                    # (1.5)
        m.addConstrs((l[t] - l_ref[t] - d_pos[t] + d_neg[t] == 0 for t in T), name=f"split_{i}")       # (2.2)
        m.addConstrs((-d_pos[t] <= 0 for t in T), name=f"dpos_lo_{i}")                                 # (2.3)
        m.addConstrs((-d_neg[t] <= 0 for t in T), name=f"dneg_lo_{i}")                                 # (2.3)
        m.addConstr(E_min - gp.quicksum(l[t] for t in T) <= 0, name=f"E_min_{i}")                      # (3.2)
        m.addConstr(s[1] - S_init - eta_ch * p_ch[1] + p_dis[1] / eta_dis == 0, name=f"soc_1_{i}")     # (3.7)
        m.addConstrs((s[t] - s[t - 1] - eta_ch * p_ch[t] + p_dis[t] / eta_dis == 0 for t in T if t > 1),
                     name=f"soc_{i}")                                                                   # (3.8)
        m.addConstrs((p_ch[t] - P_ch_bar <= 0 for t in T), name=f"ch_up_{i}")                          # (3.9)
        m.addConstrs((p_dis[t] - P_dis_bar <= 0 for t in T), name=f"dis_up_{i}")                       # (3.9)
        m.addConstrs((s[t] - S_bar <= 0 for t in T), name=f"s_up_{i}")                                 # (3.9)
        m.addConstrs((-p_ch[t] <= 0 for t in T), name=f"ch_lo_{i}")                                    # (3.10)
        m.addConstrs((-p_dis[t] <= 0 for t in T), name=f"dis_lo_{i}")                                  # (3.10)
        m.addConstrs((-s[t] <= 0 for t in T), name=f"s_lo_{i}")                                        # (3.10)
        m.addConstr(S_init - s[T[-1]] <= 0, name=f"s_end_{i}")                                         # (3.11)

    def solve(self):
        """Solve the household's own model (R, I). In C, the community model is solved instead."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{self.m.ModelName}: Gurobi status {self.m.Status}")

    def results(self):
        """After solving: (cost, hourly).
        cost    dict [DKK/day]: C (procurement; in C only the PV cost), P (penalty), D_L (disutility), total
        hourly  DataFrame per hour [kWh]: load l, PV p_PV, import p_im, export p_ex, battery p_ch / p_dis / s,
                exceedance e above the own cap (model I; 0 otherwise)
        """
        T = self.T
        hourly = pd.DataFrame({
            "l": [self.l[t].X for t in T],
            "p_PV": [self.p_PV[t].X for t in T],
            "p_im": [self.p_im[t].X for t in T],
            "p_ex": [self.p_ex[t].X for t in T],
            "p_ch": [self.p_ch[t].X for t in T],
            "p_dis": [self.p_dis[t].X for t in T],
            "s": [self.s[t].X for t in T],
        }, index=pd.Index(T, name="t"))
        if self.contract is None:
            hourly["e"] = 0.0
        else:
            hourly["e"] = [self.e[t].X for t in T]
        cost = {"C": self.C.getValue(), "P": self.P.getValue(), "D_L": self.D.getValue()}
        cost["total"] = cost["C"] + cost["P"] + cost["D_L"]
        return cost, hourly


def collect(consumers):
    """Results of several solved households, in two tables:
    members  one row per household: archetype and its cost C, P, D_L, total [DKK/day]
    hourly   one row per household and hour (index: member, t): the columns of results()
    """
    rows, hours = [], []
    for c in consumers:
        cost, hourly = c.results()
        cost["member"] = c.data.consumer_id
        cost["archetype"] = c.data.archetype
        rows.append(cost)
        hourly["member"] = c.data.consumer_id
        hours.append(hourly)
    members = pd.DataFrame(rows).set_index("member")
    hourly = pd.concat(hours).reset_index().set_index(["member", "t"])
    return members, hourly
