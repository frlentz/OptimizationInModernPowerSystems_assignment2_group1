"""Model C - community contract, central dispatch (LP): the manager dispatches all households in ONE optimization.

Every household keeps its FlexibleConsumer constraints (added to the community's model); its p_im / p_ex are now
exchanges with the community. New, at the connection point (every variable unbounded, every bound explicit):

    min  sum_t (pi_DA_t + (1-beta) tau_imp) P_imp_t - (pi_DA_t - (1-beta) tau_exp) P_exp_t + alpha E_t
         + sum_i sum_t [ c_PV p_PV_it + c_L_i (d_pos_it + d_neg_it) ]
    s.t. the constraints of every household i                                                (1.3)-(3.11), (A2.1)
         sum_i (p_im_it - p_ex_it) - P_imp_t + P_exp_t = 0      (lambda_t: internal price)    (A2.4)
         P_imp_t - Pbar_t - E_t <= 0                             (gamma_t: value of more cap)  (A2.5)
         -E_t <= 0,  -P_imp_t <= 0,  -P_exp_t <= 0                                           (A2.6)

KKT multiplier = -Pi (Gurobi's dual), as in Assignment 1.

limit_case=True (validation, Table 12 of the assignment): no cap, no discount, and the full tariffs charged on each
household's own p_im / p_ex instead of at the connection point -> must give exactly the cost of model R.

    python -m src.q1.model_C          # run the checks of model C on the base case (d1)
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from src.q1.consumer import FlexibleConsumer, collect
from src.data_loader import load_community


class CommunityModel:

    def __init__(self, com, limit_case=False):
        self.com = com
        self.limit_case = limit_case
        self.T = range(1, com.members[0].n_hours + 1)       # hours t = 1, ..., 24
        self.m = gp.Model("community_C")
        self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        com, m, T = self.com, self.m, self.T
        k = com.contract

        # Input parameters
        if self.limit_case:
            tariff_factor = 0.0                             # validation: no tariff at the meter
        else:
            tariff_factor = 1.0 - k.tariff_discount_ratio  # (1 - beta) = 0.25
        pi_DA = {t: com.energy_price[t - 1] for t in T}
        pi_imp = {t: pi_DA[t] + tariff_factor * com.import_tariff for t in T}
        pi_exp = {t: pi_DA[t] - tariff_factor * com.export_tariff for t in T}
        P_bar = {t: k.capacity_limit_kW[t - 1] for t in T}
        alpha = k.penalty_DKK_per_kWh

        # Households: their variables and constraints go into the community's model
        self.members = [FlexibleConsumer(d, model=m) for d in com.members]

        # Variables at the connection point (kWh in hour t)
        P_imp = self.P_imp = m.addVars(T, lb=-GRB.INFINITY, name="P_imp")
        P_exp = self.P_exp = m.addVars(T, lb=-GRB.INFINITY, name="P_exp")
        E = self.E = m.addVars(T, lb=-GRB.INFINITY, name="E")

        # Objective: the bill at the meter (energy + tariffs), the penalty, and every household's PV cost + disutility
        self.bill = gp.quicksum(pi_imp[t] * P_imp[t] - pi_exp[t] * P_exp[t] for t in T)
        self.P = gp.quicksum(alpha * E[t] for t in T)
        objective = self.bill + gp.quicksum(c.C + c.D for c in self.members)
        if self.limit_case:     # validation: full tariffs on each household's own flows, no penalty
            objective += gp.quicksum(com.import_tariff * c.p_im[t] + com.export_tariff * c.p_ex[t]
                                     for c in self.members for t in T)
        else:
            objective += self.P
        m.setObjective(objective, GRB.MINIMIZE)

        # Constraints, written as h(x) = 0 or g(x) <= 0
        self.balance = m.addConstrs(
            (gp.quicksum(c.p_im[t] - c.p_ex[t] for c in self.members) - P_imp[t] + P_exp[t] == 0 for t in T),
            name="balance")                                                                        # (A2.4)
        if not self.limit_case:
            self.cap = m.addConstrs((P_imp[t] - P_bar[t] - E[t] <= 0 for t in T), name="cap")     # (A2.5)
        m.addConstrs((-E[t] <= 0 for t in T), name="E_lo")                                         # (A2.6)
        m.addConstrs((-P_imp[t] <= 0 for t in T), name="P_imp_lo")                                 # (A2.6)
        m.addConstrs((-P_exp[t] <= 0 for t in T), name="P_exp_lo")                                 # (A2.6)

    def solve(self):
        """Solve; return (total, meter, members, hourly).
        total    community cost [DKK/day]: dict with total, bill (energy + tariffs at the meter), P (penalty)
        meter    DataFrame per hour at the connection point: P_imp, P_exp, E, cap [kW]; lambda, gamma [DKK/kWh]
        members, hourly   the two tables of consumer.collect() (a household's C is only its PV cost here)
        """
        m, T = self.m, self.T
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{m.ModelName}: Gurobi status {m.Status}")

        meter = pd.DataFrame({
            "P_imp": [self.P_imp[t].X for t in T],
            "P_exp": [self.P_exp[t].X for t in T],
            "E": [self.E[t].X for t in T],
            "cap": self.com.contract.capacity_limit_kW,
            "lambda": [-self.balance[t].Pi for t in T],          # internal price
        }, index=pd.Index(T, name="t"))
        if self.limit_case:
            meter["gamma"] = 0.0
        else:
            meter["gamma"] = [-self.cap[t].Pi for t in T]         # value of one more kW of cap

        total = {"total": m.ObjVal, "bill": self.bill.getValue(), "P": self.P.getValue()}
        members, hourly = collect(self.members)
        return total, meter, members, hourly


if __name__ == "__main__":
    from src.q1.model_R_I import solve_members

    com = load_community("base", day="d1")
    members_R, _ = solve_members(com)
    members_I, _ = solve_members(com, contract=com.contract)
    total_C, meter, _, _ = CommunityModel(com).solve()
    total_limit, _, _, _ = CommunityModel(com, limit_case=True).solve()

    cost_R, cost_I = members_R["total"].sum(), members_I["total"].sum()
    print("Checks of model C (base case, d1)")
    print(f"  limit case = R:  {total_limit['total']:.4f} = {cost_R:.4f} DKK")
    print(f"  C <= I:          {total_C['total']:.2f} <= {cost_I:.2f} DKK")
    both = pd.concat([meter["P_imp"], meter["P_exp"]], axis=1).min(axis=1).max()
    print(f"  never imports and exports in the same hour: largest overlap {both:.2f} kWh")
