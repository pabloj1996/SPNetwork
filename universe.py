import pickle
import numpy as np
import pandas as pd
import datetime
import scipy.stats as st
import os.path
import matplotlib.pyplot as plt


class Company:
    def __init__(self, name, df):
        self.name = name
        self.data = df
        aux = df.index.values.tolist()
        self.first_date = aux[0]
        self.last_date = aux[-1]
        self.datapoints = len(aux)
        self.data['Inc'] = self.data['Px']/self.data['Px'].shift(+1)-1

    def calculate_decorrelated_prices(self, bmk, rolling_days=100):
        inclist = self.data['Inc'].tolist()
        bmkinclist = bmk.data['Inc'].tolist()
        c = rolling_days+1
        beta_list = [np.nan for i in range(rolling_days)]
        decorr_incs = [np.nan for i in range(rolling_days)]
        decorr_p = [np.nan for i in range(rolling_days-1)]
        decorr_p.append(100.0)
        while c <= self.datapoints:
            x_ = bmkinclist[c - rolling_days:c]
            y_ = inclist[c - rolling_days:c]
            slope, intercept, r_value, p_value, std_err = st.linregress(x_, y_)
            # Slope is beta
            beta_list.append(slope)
            decorr_incs.append(inclist[c-1]/slope-bmkinclist[c-1])
            decorr_p.append(decorr_p[-1]*(1+decorr_incs[-1]))
            c += 1
        self.data['Beta'] = beta_list
        self.data['Decorr_Incs'] = decorr_incs
        self.data['Decorr_Px'] = decorr_p


class Universe:
    def __init__(self, companies, names, only_max_len = True):
        # Load companies
        self.list_of_companies = []
        for c, comp in enumerate(companies):
            if only_max_len:
                if len(comp) == 2345:
                    name = names[c]
                    C = Company(name=name, df=comp)
                    self.list_of_companies.append(C)
                else:
                    continue
            else:
                name = names[c]
                C = Company(name=name, df=comp)
                self.list_of_companies.append(C)
        # Load bmk
        bmk = load_SP500()
        Bmk = Company(name='SP500', df=bmk)
        self.benchmark = Bmk

    def add_decorrelated_prices(self, correlation_learning_days=100):
        # Iterate through companies
        c = 0
        m = len(self.list_of_companies)
        for comp in self.list_of_companies:
            c = c+1
            print('Calculating decorr. prices...', 100*c/m)
            comp.calculate_decorrelated_prices(bmk=self.benchmark, rolling_days=correlation_learning_days)


def load_SP500():
    df = pd.read_csv('SP500.csv', index_col='Date')
    inx = df.index.values.tolist()
    new_inx = []
    for i in inx:
        datetime_object = datetime.datetime.strptime(i, '%Y-%m-%d')
        new_inx.append(datetime_object)
    df['Dates'] = new_inx
    df = df.set_index('Dates')
    df['Px'] = df['Close']
    df = df[['Px']]
    return df


def load_universe():
    # Load preshaped files
    with open('companies', 'rb') as handle:
        companies = pickle.load(handle)

    with open('names', 'rb') as handle:
        names = pickle.load(handle)

    # Convert loaded data as universe class
    Uni = Universe(companies=companies, names=names)
    return Uni

# To save time this was stored in a file since the last run, so it is loaded through
def build_data_structure():
    if os.path.exists('Universe_100dBeta'):
        print('Loading stored data...')
        with open('Universe_100dBeta', 'rb') as handle:
            uni_100d = pickle.load(handle)
        print('Load successful!')

    else:
        print('File not found so data structure is being built...')
        # Load and calculate decorrelated series
        uni_100d = load_universe()
        rolling_beta_days = 100
        uni_100d.add_decorrelated_prices(correlation_learning_days=rolling_beta_days)

        with open('Universe_100dBeta', 'wb') as handle:
            pickle.dump(uni_100d, handle)
        print('Load successful!')

    return uni_100d


uni = build_data_structure()