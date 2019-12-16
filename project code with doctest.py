from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# get the file name in directory  (world happiness)
def fetch_dataset_file(file_dir):
    """
    Function to read in filename
    Argument: file_dir: file directory
    Return:
            k:the list of files name
    >>> res=fetch_dataset_file("world-happiness")
    >>> res
    ['2015.csv', '2016.csv', '2017.csv']
    """
    for i, j, k in os.walk(file_dir):
        return k

# merge the dataset  (world happiness)
def merge_dataset(file_dir, file_name, merge_on):
    """
    Function to merge several datasets
    Argument:
            fileDirectory: file directory
            file_name:the list of files name
            merge_on:the attribute that merged
    Return:
            data: merged dataframe

    >>> file_happiness = ["2015.csv","2016.csv","2017.csv"]
    >>> data=merge_dataset('world-happiness',file_happiness, 'Country')
    >>> data.shape
    (146, 35)
    """
    tmp = pd.read_csv(file_dir + '//' + file_name[0])
    data = tmp[tmp.columns.tolist()[1:]].add_prefix(file_name[0].split('.')[0] + '_')  # add prefix to columns
    data.insert(0, merge_on, tmp[merge_on])  # add merged attribute column
    for i in range(1, len(file_name)):
        tmp = pd.read_csv(file_dir + '//' + file_name[i])
        d_tmp = tmp[tmp.columns.tolist()[1:]].add_prefix(file_name[i].split('.')[0] + '_')
        d_tmp.insert(0, merge_on, tmp[merge_on])
        data = pd.merge(data, d_tmp, left_on=merge_on, right_on=merge_on)  # merge dataset
    data.dropna(axis=0, how='any', inplace=True)  # drop nal values
    return data

# find the average happiness score of each country in three years
def country_sort(happiness_data):
    """
    Function to sort countries by average happies scores in three years
    Argument:
            happiness_data: the data of world happiness
    Return:
            h_d: the data of each country's average happiness score in three years

    >>> data=merge_dataset('world-happiness', ['2015.csv',"2016.csv","2017.csv"], 'Country')
    >>> res=country_sort(data)
    >>> res.shape
    (146, 2)
    """
    score = (happiness_data['2015_Happiness Score'] + happiness_data[
        '2016_Happiness Score']  # calculate average happiness scores
             + happiness_data['2017_Happiness.Score']) / 3
    h_d = DataFrame({'Country': happiness_data['Country'], 'Avg_Happiness Score': score})
    h_d.sort_values("Avg_Happiness Score", inplace=True)  # sort dataset
    return h_d


# fetch attributes
def fetch_attributes(dataset, targetvalue, attribute, target):
    """
    Function to select specific rows and columns
    Return:
            dataset: dataframe which fetch attributes
    >>> NFA_df = pd.read_csv('NFA 2018.csv')  # read ecological footprints datase
    >>> nfa_df = fetch_attributes(NFA_df, ['EFConsPerCap'], ['country', 'record', 'carbon', 'total', 'year'],'record')
    >>> nfa_df.shape
    (8702, 5)
    """
    return dataset.loc[dataset[target].isin(targetvalue)][attribute]


# select corresponding year (ecological footprints and biocapacity)
def select_year(dataset, start_year, end_year):
    """
    Function to select years from dataset
    Argument:
            dataset:processing dataset
            start_year:select year from
            end_year:select year ended
    Return:
            data: the data between start year and end year
    >>> NFA_df = pd.read_csv('NFA 2018.csv')
    >>> c_nfa = select_year(NFA_df, 2013, 2014)
    >>> c_nfa.shape
    (3660, 5)
    """
    country = []
    dataset = dataset.loc[dataset['year'].isin(range(start_year, end_year + 1))]  # data from start year to end year
    tmp = pd.DataFrame(dataset.groupby([dataset['country'], dataset['record']]).size())
    tmp = tmp[tmp.values == (end_year - start_year + 1)] #select the entries which have records from the start year to end year
    for i in tmp.index.tolist():
        country.append(i[0])
    data = fetch_attributes(dataset, country, ['country', 'record', 'carbon', 'total', 'year'], 'country')
    return data

def f_3(x, A, B, C, D):                                                                       #cubic equation to fit data
    """
    Function to calculate cubic equation
        Argument:
            x:independent variable
            A:cubic term parameter
            B:quadratic term parameter
            C:primary term parameter
            D:constant parameter
        Return:
            the result of cubic equation
        >>> res=f_3(1, 1, 2, 3, 4)
        >>> res
        10
    """
    return A*x*x*x + B*x*x + C*x + D
def plot_test(x0,y0):                                                                        #fit data with curve
    """
        Function to fit curve in relationship between Ecological Footprint per capital and Happies score
        Argument:
            x0:independent variable
            y0:dependent variable
        Return:
            show scatter plot
        """
    plt.figure(figsize=(10,7)) #set the figure size
    plt.scatter(x0[:], y0[:], 25, "blue")
    A3, B3, C3, D3= curve_fit(f_3, x0, y0)[0]
    x3 = np.arange(0, 6, 0.01)
    y3 = A3*x3*x3*x3 + B3*x3*x3 + C3*x3 + D3
    plt.plot(x3, y3, "red")
    plt.title("Relationship between Ecological Footprint per capital and Happies score")
    plt.xlabel('EFConsPerCap')
    plt.ylabel('Avg_Happiness Score')
    plt.show()
    return

def healthyplot(x,y):
    """
        Function to fit curve in relationship between Ecological Footprint per capital and Happies score
        Argument:
            x:independent variable
            y:dependent variable
        Return:
            show scatter plot
    """
    X = h_df[x].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = h_df[y].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y,color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.show()
    return



if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)
    #hypothesis 1
    file_happiness = fetch_dataset_file("world-happiness")  # read world happiness datasets
    h_df = merge_dataset('world-happiness', file_happiness, 'Country')
    NFA_df = pd.read_csv('NFA 2018.csv')  # read ecological footprints datase
    NFA_df.dropna(axis=0, how='any', inplace=True)
    nfa_df = fetch_attributes(NFA_df, ['EFConsPerCap'], ['country', 'record', 'carbon', 'total', 'year'],
                              'record')  # only select ecological footprints per capita

    h_df_15_17 = h_df[['Country', '2015_Happiness Score', '2016_Happiness Score',
                       '2017_Happiness.Score']]  # select three years' average happiness score
    gdpdf = pd.read_csv('WEODataGDP.CSV')  # read world GDP dataset
    gdp_h_df_15_17 = pd.merge(gdpdf, h_df_15_17, left_on='Country', right_on='Country',
                              how='inner')  # merage dataset with world happiness
    gdp_h_df_15_17.dropna(axis=0, how='any', inplace=True)  # drop nal values
    gdp_h_df_15_17 = gdp_h_df_15_17[
        gdp_h_df_15_17['Subject Descriptor'] == 'Gross domestic product per capita, current prices']

    plt.figure(figsize=(10, 7))  # scatter plot which show relationship between GDP and happiness
    l1 = plt.scatter(gdp_h_df_15_17['2015'], gdp_h_df_15_17['2015_Happiness Score'], color='red', label='2015')
    l2 = plt.scatter(gdp_h_df_15_17['2016'], gdp_h_df_15_17['2016_Happiness Score'], color='blue', label='2016')
    l2 = plt.scatter(gdp_h_df_15_17['2017'], gdp_h_df_15_17['2017_Happiness.Score'], color='green', label='2017')
    plt.title('GDP with happies', size=24)
    plt.xlabel('GDP', size=18)
    plt.ylabel('happies score', size=18)
    plt.legend(loc=2, prop={'size': 18})
    plt.show()

    cpidf = pd.read_csv('WEODataCPI.CSV')  # read world CPI dataset
    cpi_h_df_15_17 = pd.merge(cpidf, h_df_15_17, left_on='Country', right_on='Country',
                              how='inner')  # merage dataset with world happiness
    cpi_h_df_15_17.dropna(axis=0, how='any', inplace=True)  # drop nal values

    plt.figure(figsize=(10, 7))  # scatter plot which show relationship between CPI and happiness

    l1 = plt.scatter(cpi_h_df_15_17['2015'], cpi_h_df_15_17['2015_Happiness Score'], color='red', label='2015')
    l2 = plt.scatter(cpi_h_df_15_17['2016'], cpi_h_df_15_17['2016_Happiness Score'], color='blue', label='2016')
    l2 = plt.scatter(cpi_h_df_15_17['2017'], cpi_h_df_15_17['2017_Happiness.Score'], color='green', label='2017')
    plt.title('CPI with happies', size=24)
    plt.xlabel('CPI', size=18)
    plt.ylabel('happies score', size=18)
    plt.legend(loc=2, prop={'size': 18})
    plt.xlim(0, 1000)
    plt.show()

    # hypothesis 2
    c_nfa = fetch_attributes(nfa_df, nfa_df['country'], ['country', 'record', 'carbon', 'total', 'year'], 'country')
    c_nfa = select_year(c_nfa, 1994, 2014)
    c_nfa_mean = c_nfa.groupby('country')['carbon'].agg(['mean'])  # calculate average carbon emissions
    happycountries = country_sort(h_df)
    combined = pd.merge(happycountries, c_nfa_mean, left_on='Country', right_on='country', how='inner')
    plt.style.use('ggplot')  # frequency plot for carbon emission
    combined['mean'].plot(kind='hist', color='blue', edgecolor='black', figsize=(10, 7))
    plt.title('Distribution of carbon emission', size=24)
    plt.xlabel('Carbon emssion', size=18)
    plt.ylabel('Frequency', size=18)
    # Regression plot using seaborn.
    fig = plt.figure(figsize=(10, 7))
    sns.regplot(x=combined['mean'], y=combined['Avg_Happiness Score'], color='blue', marker='+')
    # Legend, title and labels.
    plt.legend(labels='carbon')
    plt.title('Relationship between EFExportsPerCap and happies', size=24)
    plt.xlabel('EFExportsPerCap', size=18)
    plt.ylabel('happies', size=18)
    plot_test(combined['mean'], combined['Avg_Happiness Score'])
    # hypothesis 3
    healthyplot('2015_Happiness Score', '2015_Health (Life Expectancy)')  # linear regression using data in 2015
    healthyplot('2016_Happiness Score', '2016_Health (Life Expectancy)')  # linear regression using data in 2016
    healthyplot('2017_Happiness.Score', '2017_Health..Life.Expectancy.')  # linear regression using data in 2017
