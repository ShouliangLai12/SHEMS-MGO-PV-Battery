# %%
# %%
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mealpy import FloatVar, Problem
from mealpy import GWO, MGO, PSO, GA
import os
mg_rate = 24
def const_to_array(price, rate=24):
    if hasattr(price, "__len__"):
        price = price
    else:
        price = np.ones(rate) * price
    return price
def env_limiting(power, power_polution, environmental_limits=None):
    if environmental_limits:
        power_limits = min(np.array(list(environmental_limits.values()))/np.array(list(power_polution.values())))
        for i in range(len(power)):
            if power[i] > power_limits:
                power[i] = power_limits
            else:
                pass
    return power
def add_unavailability(power_array, uncertainty_percent=[0.2]*mg_rate):
    if hasattr(uncertainty_percent, "__len__"):
        uncertainty_percent = uncertainty_percent
    else:
        uncertainty_percent = np.ones(mg_rate) * uncertainty_percent
    unavailability_hour=range(mg_rate)
    power_array = np.array(power_array)
    for i, u in enumerate(unavailability_hour):
        p = power_array[u]
        power_array[u] = p + (p * (uncertainty_percent[i]))
    return power_array
class Microgrid:
    def __init__(self, load):
        self.load = load
        self.components_power = dict()
        self.components_price = dict()
        self.compulsory_min = dict()
        self.environmental = dict()
        self.environmental_limits = None
        self.nondispatchable = dict()
        self.log = []
    def PV_product(self, power=None, price=None, uncertainty=0.0):
        self.environmental['PV'] = {'co2':0.0, 'so2':0.0, 'nox':0.0}
        power_with_uncertainty = add_unavailability(const_to_array(power), uncertainty_percent=uncertainty)
        self.components_power['PV'] = env_limiting(power_with_uncertainty, self.environmental['PV'], self.environmental_limits)
        self.components_price['PV'] = const_to_array(price)
        self.compulsory_min['PV'] = 0
        self.nondispatchable['PV'] = env_limiting(power_with_uncertainty, self.environmental['PV'], self.environmental_limits)
    def Grid_product(self, power=None, sellprice=None, buyprice=None):
        self.grid_min = 0
        self.grid_max = power
        self.environmental['Grid'] = {'co2':0.3*0.95, 'so2':0.3*3.5e-4, 'nox':0.3*2.0e-4}
        self.components_power['Grid'] = env_limiting(const_to_array(power), self.environmental['Grid'], self.environmental_limits)
        self.components_price['Grid'] = const_to_array(buyprice)
        self.compulsory_min['Grid'] = 0
        self.grid_sellprice = const_to_array(sellprice)
    def Battery_product(self, price = 0.0, price2 = 0.35, sto_loss = 0.9, capacity = 20, rate = 4, sto_min=0.0):
        self.sto_rate = rate
        self.sto_loss = sto_loss
        self.sto_cap = capacity
        self.sto_min = sto_min
        self.sto_price = price
        self.sto_price2 = price2
        self.sto_state = 0.0
        self.environmental['Battery'] = {'co2':0.005, 'so2':1.5e-7, 'nox':1.0e-6}
        self.components_power['Battery'] = env_limiting(const_to_array(rate), self.environmental['Battery'], self.environmental_limits)
        self.components_price['Battery'] = const_to_array(price)
        self.compulsory_min['Battery'] = 0
    def EV(self, EV_drive ,EV_price = 0.02, EV_loss = 0.9, EV_capacity = 20, EV_rate = 4, EV_min=0.0):
        self.EV_rate = EV_rate
        self.EV_loss = EV_loss
        self.EV_cap = EV_capacity
        self.EV_min = EV_min
        self.EV_price = EV_price
        self.EV_state = 0.0
        self.EV_trip = list(range(np.where(EV_drive>0)[0][0], np.where(EV_drive>0)[0][-1]+1))
        self.environmental['EV'] = {'co2':0.005, 'so2':1.5e-7, 'nox':1.0e-6}
        self.components_power['EV'] = env_limiting(const_to_array(np.zeros(mg_rate)), self.environmental['EV'], self.environmental_limits)
        self.EV_drive = EV_drive
        self.components_price['EV'] = const_to_array(EV_price)
        self.compulsory_min['EV'] = 0
def Charging(mg, psto, kind):
    if kind == 'Battery':
        sto_init = mg.sto_min
        sto_loss = mg.sto_loss
        capacity = mg.sto_cap
        rate = mg.sto_rate
        sto_min = -mg.sto_min*mg.sto_cap
        sto_stat = mg.sto_state
    else:
        sto_init = mg.EV_min
        sto_loss = mg.EV_loss
        capacity = mg.EV_cap
        rate = mg.EV_rate
        sto_min = -mg.EV_min*mg.EV_cap
        sto_stat = mg.EV_state
    if sto_stat > 0:
        sto_stat = -1 * sto_stat 
    if psto < 0:
        if abs(psto)> rate:
            psto = rate*-1
        if abs(sto_stat) <= capacity-(rate*sto_loss):
            if abs(psto) >= rate*sto_loss:
                storage = rate*sto_loss
            else:
                storage = abs(psto)*sto_loss
            sto_stat += -1*storage
        elif abs(sto_stat) > capacity-(rate*sto_loss) and abs(sto_stat) <= capacity:
            if abs(psto) >= (capacity-abs(sto_stat)):
                storage = (capacity-abs(sto_stat))
            else:
                storage = abs(psto)
            sto_stat += -1*storage
        else:
            storage = 0.0
    else:
        storage = 0.0
    if kind == 'Battery':
        mg.sto_state = sto_stat
    else:
        mg.EV_state = sto_stat

    return -1*storage/sto_loss

def Discharge(mg, psto, kind):
    if kind == 'Battery':
        sto_init = mg.sto_min
        sto_loss = mg.sto_loss
        capacity = mg.sto_cap
        rate = mg.sto_rate
        sto_min = -mg.sto_min*mg.sto_cap
        sto_stat = mg.sto_state
    else:
        sto_init = mg.EV_min
        sto_loss = mg.EV_loss
        capacity = mg.EV_cap
        rate = mg.EV_rate
        sto_min = -mg.EV_min*mg.EV_cap
        sto_stat = mg.EV_state
    if psto > 0:
        if abs(sto_stat)-abs(sto_min) >= psto:
            if abs(psto) >= rate:
                storage = rate
            else:
                storage = abs(psto)
            sto_stat += storage
        elif abs(sto_stat)-abs(sto_min) < psto:
            if abs(psto) >= abs(sto_stat)-abs(sto_min):
                storage = abs(sto_stat)-abs(sto_min)
            else:
                storage = abs(psto)
            sto_stat += storage
        else:
            storage = 0.0
    else:
        storage = 0.0
    if kind == 'Battery':
        mg.sto_state = sto_stat
    else:
        mg.EV_state = sto_stat
    return storage*sto_loss            
        
def dispatch(mg: Microgrid, k):
    pload = mg.load[k]
    psto = mg.components_power['Battery'][k]
    if 'EV' in list(mg.components_power.keys()):
        if k in mg.EV_trip:
            pEV = mg.EV_drive[k]/mg.EV_loss
        else:
            pEV = mg.components_power['EV'][k]
        EV_drive = mg.EV_drive[k]
    else:
        pEV = 0.0
        EV_drive = 0.0
    components = sorted([[comp[0], comp[1][k]] for comp in mg.components_price.items()], key=lambda x: x[1], reverse=False)
    components_price = dict([[comp[0], mg.components_price[comp[0]][k]] for comp in components])
    components_power = dict([[comp[0], mg.components_power[comp[0]][k]] for comp in components])
    spend = dict(zip(list(dict(components).keys()), len(components)*[0.0]))
    log = []
    compulsory = dict([[compul[0], compul[1]] for compul in mg.compulsory_min.items()])
    for compul in compulsory:
        power = compulsory[compul]  
        if power>0:
            cmp_val = min(pload, power)
            spend[compul] += cmp_val
            log.append([k, compul, 'Load', cmp_val])
            pload = pload - cmp_val
            power = power - cmp_val
            components_power[compul] = components_power[compul] - cmp_val
            if psto < 0:
                if components_price['Battery'] <= components_price[compul] and power>0:
                    ch_val = min(abs(psto), abs(pload))
                    storage = Charging(mg, ch_val*-1, 'Battery')
                    spend[compul] += abs(storage)
                    spend['Battery'] += storage
                    log.append([k, compul, 'Battery', abs(storage)])
                    psto += abs(storage)
                    power = power - abs(storage)
                    components_power[compul] = components_power[compul] - ch_val
            if pEV < 0:
                if components_price['EV'] <= components_price[compul] and power>0:
                    ch_val = min(abs(pEV), power)
                    storage = Charging(mg, ch_val*-1, 'EV')
                    spend[compul] += abs(storage)
                    spend['EV'] += storage
                    log.append([k, compul, 'EV', abs(storage)])
                    pEV += abs(storage)
                    power = power - abs(storage)
                    components_power[compul] = components_power[compul] - ch_val
            if components_price[compul] <= mg.grid_sellprice[k] and power>0:
                spend[compul] += power
                spend['Grid'] += -1 * power
                log.append([k, compul, 'Grid', power])
                components_power[compul] = components_power[compul] - power
                power = 0.0        
    for nondiscomp in list(mg.nondispatchable.keys()):
        power = components_power[nondiscomp] 
        if power > 0:        
            if pload > 0:
                cmp_val = min(pload, power)
                pload = pload - cmp_val
                spend[nondiscomp] = cmp_val
                log.append([k, nondiscomp, 'Load', cmp_val])
                power = power - cmp_val
                if psto < 0:
                    if components_price['Battery'] <= components_price[nondiscomp]  and power>0:
                        ch_val = min(abs(psto), power)
                        storage = Charging(mg, ch_val*-1, 'Battery')
                        spend[nondiscomp] += abs(storage)
                        spend['Battery'] += storage
                        log.append([k, nondiscomp, 'Battery', abs(storage)])
                        psto += abs(storage)
                        power = power - abs(storage)
                if pEV < 0:
                    if components_price['EV'] <= components_price[nondiscomp]  and power>0:
                        ch_val = min(abs(pEV), power)
                        storage = Charging(mg, ch_val*-1, 'EV')
                        spend[nondiscomp] += abs(storage)
                        spend['EV'] += storage
                        log.append([k, nondiscomp, 'EV', abs(storage)])
                        pEV += abs(storage)
                        power = power - abs(storage)
                if components_price[nondiscomp] <= mg.grid_sellprice[k] and power>0:
                    spend[nondiscomp] += power
                    spend['Grid'] += -1 * power
                    log.append([k, nondiscomp, 'Grid', power])
                    power = 0
            else:
                if psto < 0:
                    if components_price['Battery'] <= components_price[nondiscomp]  and power>0:
                        ch_val = min(abs(psto), power)
                        storage = Charging(mg, ch_val*-1, 'Battery')
                        spend[nondiscomp] += abs(storage)
                        spend['Battery'] += storage
                        log.append([k, nondiscomp, 'Battery', abs(storage)])
                        psto += abs(storage)
                        power = power - abs(storage)
                if pEV < 0:
                    if components_price['EV'] <= components_price[nondiscomp]  and power>0:
                        ch_val = min(abs(pEV), power)
                        storage = Charging(mg, ch_val*-1, 'EV')
                        spend[nondiscomp] += abs(storage)
                        spend['EV'] += storage
                        log.append([k, nondiscomp, 'EV', abs(storage)])
                        pEV += abs(storage)
                        power = power - abs(storage)
                if components_price[nondiscomp] <= mg.grid_sellprice[k] and power>0:
                    spend[nondiscomp] += power
                    spend['Grid'] += -1 * power
                    log.append([k, nondiscomp, 'Grid', power])
    dispatchables = dict()
    for cmp in list(components_power.keys()):
        if cmp not in list(mg.nondispatchable.keys()):
            dispatchables[cmp] = components_power[cmp]
    components_power = dispatchables
    for cmp in list(components_power.keys()):
        power = components_power[cmp]
        if cmp == 'EV':
            if EV_drive > 0:
                dc_val = min(EV_drive, pEV*mg.EV_loss)
                storage = Discharge(mg, dc_val/mg.EV_loss, 'EV')
                EV_trip = storage
                EV_drive = EV_drive - abs(storage)
                power = power - storage/mg.EV_loss
                log.append([k, cmp, 'Trip', EV_trip])
            else:
                EV_trip = 0
        if power > 0:
            if pload > 0:
                if cmp == 'Battery':
                    if (psto > 0) and (abs(mg.sto_state) > 0):
                        dc_val = min(pload, psto*mg.sto_loss)
                        storage = Discharge(mg, dc_val/mg.sto_loss, 'Battery')
                        spend[cmp] += storage
                        log.append([k, cmp, 'Load', storage])
                        pload = pload - abs(storage)
                        power = power - storage/mg.sto_loss
                        if components_price[cmp] <= mg.grid_sellprice[k] and power>0:
                            storage = Discharge(mg, power, 'Battery')
                            spend[cmp] += storage
                            spend['Grid'] += -1 * abs(storage)
                            log.append([k, cmp, 'Grid', storage])
                elif cmp == 'EV':
                    if (pEV > 0) and (abs(mg.EV_state) > 0):
                        dc_val = min(pload, pEV*mg.EV_loss)
                        storage = Discharge(mg, dc_val/mg.EV_loss, 'EV')
                        spend[cmp] += storage
                        log.append([k, cmp, 'Load', storage])
                        pload = pload - abs(storage)
                        power = power - storage/mg.EV_loss
                        if components_price[cmp] <= mg.grid_sellprice[k] and power>0:
                            storage = Discharge(mg, power, 'EV')
                            spend[cmp] += storage
                            spend['Grid'] += -1 * abs(storage)
                            log.append([k, cmp, 'Grid', storage])
                else:
                    cmp_val = min(pload, power)
                    spend[cmp] += cmp_val
                    log.append([k, cmp, 'Load', cmp_val])
                    pload = pload - cmp_val
                    power = power - cmp_val
                    if psto < 0:
                        if components_price['Battery'] <= components_price[cmp] and power>0:
                            ch_val = min(abs(psto), power)
                            storage = Charging(mg, ch_val*-1, 'Battery')
                            spend[cmp] += abs(storage)
                            spend['Battery'] += storage
                            log.append([k, cmp, 'Battery', abs(storage)])
                            psto += abs(storage)
                            power = power - abs(storage)
                    if pEV < 0:
                        if components_price['EV'] <= components_price[cmp] and power>0:
                            ch_val = min(abs(pEV), power)
                            storage = Charging(mg, ch_val*-1, 'EV')
                            spend[cmp] += abs(storage)
                            spend['EV'] += storage
                            log.append([k, cmp, 'EV', abs(storage)])
                            pEV += abs(storage)
                            power = power - abs(storage)
                    if cmp != 'Grid':
                        if components_price[cmp] <= mg.grid_sellprice[k] and power>0:
                            spend[cmp] += power
                            spend['Grid'] += -1 * power
                            log.append([k, cmp, 'Grid', power])
                            power = 0.0
            else:
                if psto < 0:
                    if components_price['Battery'] <= components_price[cmp] and power>0:
                        ch_val = min(abs(psto), power)
                        storage = Charging(mg, ch_val*-1, 'Battery')
                        spend[cmp] += abs(storage)
                        spend['Battery'] += storage
                        log.append([k, cmp, 'Battery', abs(storage)])
                        psto += abs(storage)
                        power = power - abs(storage)
                if pEV < 0:
                    if components_price['EV'] <= components_price[cmp] and power>0:
                        ch_val = min(abs(pEV), power)
                        storage = Charging(mg, ch_val*-1, 'EV')
                        spend[cmp] += abs(storage)
                        spend['EV'] += storage
                        log.append([k, cmp, 'EV', abs(storage)])
                        pEV += abs(storage)
                        power = power - abs(storage)
                if cmp != 'Grid':
                    if components_price[cmp] <= mg.grid_sellprice[k] and power>0:
                        if cmp == 'Battery':
                            if (psto > 0) and (abs(mg.sto_state) > 0):
                                storage = Discharge(mg, psto, 'Battery')
                                spend[cmp] += storage
                                spend['Grid'] += -1 * abs(storage)
                                log.append([k, cmp, 'Grid', storage])
                        elif cmp == 'EV':
                            if (pEV > 0) and (abs(mg.EV_state) > 0):
                                storage = Discharge(mg, pEV, 'EV')
                                spend[cmp] += storage
                                spend['Grid'] += -1 * abs(storage)
                                log.append([k, cmp, 'Grid', storage])
                        else:       
                            spend[cmp] += power
                            spend['Grid'] += -1 * power
                            log.append([k, cmp, 'Grid', power])
    cost = dict()       
    for comp in spend:
        if spend[comp] < 0 and  comp == 'Grid':
            cost[comp] = 0.0
        else:
            cost[comp] = components_price[comp] * abs(spend[comp])
    if 'EV' in list(mg.components_power.keys()):
        cost['EV_exec'] = EV_drive * 10000
    if mg.sto_price==0.0:    
        if spend['Grid'] < 0:
            cost['grid_sell'] = -1 * ((abs(spend['Grid'])-spend['Battery'])*mg.grid_sellprice[k])
        else:
            cost['grid_sell'] = 0.0
        if (spend['Battery'] > 0):
            cost['Battery'] = -1*(spend['Battery'] * mg.sto_price2)
    else:
        if spend['Grid'] < 0:
            cost['grid_sell'] = -1 * ((abs(spend['Grid']))*mg.grid_sellprice[k])
        else:
            cost['grid_sell'] = 0.0
    cost['sum'] = sum(cost.values())
    spend['sum'] = sum(spend.values())
    spend['Battery_state'] = mg.sto_state
    if 'EV' in list(mg.components_power.keys()):
        spend['EV_trip'] = EV_trip
        if k in mg.EV_trip:
            cost['EV'] = components_price['EV'] * abs(spend['EV_trip'])
        spend['EV_state'] = mg.EV_state
    return spend, cost, log

def simulate(mg: Microgrid, x):
    battery_range = x[:mg_rate]
    K = mg_rate 
    spend_list=[]; cost_list=[]
    mg.components_power['Battery'] = np.array(battery_range)
    mg.sto_state = -mg.sto_min*mg.sto_cap
    mg.log = []
    if 'EV' in list(mg.components_power.keys()):
        EV_range = np.array(x[mg_rate:])
        EV_range[mg.EV_trip] = mg.EV_drive[mg.EV_trip]
        mg.components_power['EV'] = EV_range
        mg.EV_state = -mg.EV_min*mg.EV_cap
    for k in range(K):
        spend, cost, log = dispatch(mg, k)
        spend_list.append(spend)
        cost_list.append(cost)
        mg.log.append(log)
    output_power = pd.DataFrame(spend_list)   
    output_cost = pd.DataFrame(cost_list) 
    return output_power, output_cost
folder = 'scenario3'
schedule = 'Random' if folder == 'scenario1' else 'optimized'
section = 'summer'
load_data = pd.read_excel(f'{schedule}_schedule_{section}.xlsx', index_col=0)
data = pd.read_excel('Data.xlsx', index_col='hour')
EV_data = pd.read_excel('EV_data.xlsx', index_col='hour')['power']
mg = Microgrid(np.array(load_data['Total']))
mg.Grid_product(power=20, sellprice=np.array(data[f'grid_sell_{section}']), buyprice=np.array(data[f'grid_buy_{section}']))
mg.PV_product(power=np.array(data[f'PV_{section}']), price=0.092, uncertainty=0)
CUnit = 1
Ebatt0 = 2.0
CBOP = 33.868; CPCS = 24.695
PB0 = Ebatt0/5
n = 3000
Bcapital = CUnit * Ebatt0 + CBOP * Ebatt0 + CPCS * PB0
Blifetime = n * Ebatt0 * 2
BLCOS0 = Bcapital/Blifetime
Ebatt = 4.2
BLCOS = BLCOS0*(Ebatt/Ebatt0)**1.4285714286
PB = Ebatt/5
mg.Battery_product(price=BLCOS, price2=0.0, sto_loss=0.95, capacity=Ebatt, rate=PB, sto_min=0.0)
def obj(x):
    powers, costs = simulate(mg, x)
    return sum(costs['sum'])
if 'EV' in list(mg.components_power.keys()):
    LB = [-mg.sto_rate]*mg_rate + [-mg.EV_rate]*mg_rate
    UB = [mg.sto_rate]*mg_rate + [mg.EV_rate]*mg_rate
else:
    LB = [-mg.sto_rate]*mg_rate
    UB = [mg.sto_rate]*mg_rate
problem = {
    "obj_func": obj,
    "bounds": FloatVar(lb=LB, ub=UB),
    "minmax": "min",
    "verbose": True,
    "save_population": True}
opt_Model=GWO.OriginalGWO(epoch=500, pop_size=100)
model_name = opt_Model.name
g_best = opt_Model.solve(problem)
powers, costs = simulate(mg, g_best.solution)
print(g_best)
print(sum(costs['sum']))