import pandas as pd
from skimpy import clean_columns
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import imageio
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot

servers = pd.read_csv('certified-enterprise-servers-2023-01-12_clean.csv')
servers = clean_columns(servers)

four_socket_map = servers[servers['product_processor_socket_count']=='4']
four_socket_map = four_socket_map.iloc[:,[0,1]]
four_socket_map.to_csv('indexmap.csv',index=True)

pca = PCA(n_components=2)
four_sockets = servers[servers['product_processor_socket_count']=='4']
four_sockets = four_sockets.replace(' ','',regex = True)
four_sockets = four_sockets.replace('-','_',regex = True)
four_sockets = four_sockets.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
four_sockets.iloc[:,3:16] = four_sockets.iloc[:,3:16].astype(float)
four_sockets.iloc[:,18:95] = four_sockets.iloc[:,18:95].astype(float)
four_sockets.fillna(0,inplace = True)
four_sockets = four_sockets.applymap(lambda x: x.lower() if type(x) == str else x)
rowid = four_sockets.index
four_sockets_dum = pd.get_dummies(four_sockets, columns=['brand_name','form_factor','processor_brand_typical_or_single_configuration'] )
four_sockets_dum = four_sockets_dum.drop(four_sockets_dum.columns[0],axis=1)
four_socket_pca = pca.fit_transform(four_sockets_dum)
four_socket_pca = pd.DataFrame(four_socket_pca,index = rowid)
four_socket_pca = four_socket_pca.rename(columns = {0:'PCA1',1:'PCA2'})
four_sockets = four_sockets.join(four_socket_pca)
four_sockets['brand_name'] = four_sockets['brand_name'].astype(str)
four_sockets.loc[four_sockets['brand_name']=='dell','brand_name'] = 'dellemc'
four_sockets.loc[four_sockets['brand_name']=='hpeproliantdl560gen10','brand_name'] = 'hewlettpackardenterprise'
four_sockets.dtypes
four_sockets['total_joules']=four_sockets['power_supply_rated_output_typical_or_single_configuration_w']*four_sockets['num_power_supplies_for_typical_or_single_configuration']
four_sockets.fillna(0,inplace = True)
four_sockets = four_sockets.drop(four_sockets.columns[four_sockets.columns.str.contains('high')], axis=1)
four_sockets = four_sockets.drop(four_sockets.columns[four_sockets.columns.str.contains('low')], axis=1)
four_sockets.iloc[:,22:33] = four_sockets.iloc[:,22:33].apply(pd.to_numeric, errors='coerce', downcast='float')
four_sockets['total_joules'] = four_sockets['total_joules'].apply(pd.to_numeric, errors='coerce', downcast='float')
four_sockets_scl = four_sockets.copy()
four_sockets_scl.iloc[:,22:33] = four_sockets_scl.iloc[:,22:33].mul(four_sockets_scl['total_joules'], axis=0).astype(float)

# Embodied Emissions Stacked Bar Chart

pcf = pd.read_csv('serverCfreports2.csv')
pcf = pcf.loc[pcf['Use Emissions'] !=0]
pcf = pcf.drop(pcf.columns[[0,3]],axis = 1)
plt.rcParams["figure.figsize"] = (24, 16)
b1 = plt.barh(pcf['Brand Name']+' ' + pcf['Model Name'],pcf['Total Emissions'],color = 'lightsteelblue')
b2 = plt.barh(pcf['Brand Name']+' ' + pcf['Model Name'],pcf['Use Emissions'], color = 'blue')
plt.legend([b2, b1], ["Operational Emissions", "Embodied Emissions"], title="Emissions", loc="upper center" ,bbox_to_anchor=(0.5, 1.1),ncols=2)
plt.show()

sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)





# Grouping by Thread Count and averaging the performance
four_sockets_threads = four_sockets_scl.iloc[:,22:33].groupby(four_sockets_scl.iloc[:,10]).mean()
four_sockets_threads['Average Watts'] = four_sockets_scl.iloc[:,40].groupby(four_sockets_scl.iloc[:,10]).mean()
four_sockets_threads['CPU Average Performance'] = four_sockets_threads.iloc[:,[0,1,2,3,4,5,10]].mean(axis=1)
four_sockets_threads['Memory Average Performance'] = four_sockets_threads.iloc[:,[6,7]].mean(axis=1)
four_sockets_threads['Storage Average Performance'] = four_sockets_threads.iloc[:,[8,9]].mean(axis=1)
four_sockets_threads = four_sockets_threads.drop(four_sockets_threads.columns[0:11],axis=1)
four_sockets_threads['index'] = four_sockets_threads.index
four_sockets_threads.iloc[:,1:4] = four_sockets_threads.iloc[:,1:4].mul(four_sockets_threads['Average Watts'],axis=0)
four_sockets_threads_perf = pd.melt(four_sockets_threads,id_vars = 'index', value_vars = four_sockets_threads.iloc[:,1:4])
sns.barplot(x='index',y='value',hue='variable',data = four_sockets_threads_perf)
plt.xlabel('Thread Count', fontsize=20)
plt.ylabel('Transactions per Second', fontsize=20)
plt.legend(title = 'SERT Workload Category')

########## Plotting Workload Efficiency Scores ##########################
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
df = four_sockets.copy()
subset_columns = df.columns[df.columns.get_indexer(df.columns.values[22:32])]
for col in subset_columns:
    sns.barplot(x=df.index, y=df[col],color='gray')
    plt.show()




# Plotting the PCA Analysis
y = pd.DataFrame(four_sockets.columns)
plt.rcParams["figure.figsize"] = (12, 8)
sns.scatterplot(x ='PCA1', y ='PCA2',hue='brand_name',data = four_sockets)
plt.show()
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)

######## Server Capacity Graphs ########################################
df2 = four_sockets_scl.copy()
subset_columns = df2.columns[df2.columns.get_indexer(df2.columns.values[22:32])]
for col in subset_columns:
    colname = col.split('_eff')[0].upper()
    colname = colname.replace('_',' ')
    sns.barplot(x=df2.index, y=df2[col],color='gray').set(title=colname)
    plt.xlabel('Server Index')
    plt.ylabel('Transactions per Second')
    plt.show()

################### Emissions Rates Graphs ####################################
state_emissions = clean_columns(pd.read_excel(r'egrid2020_data.xlsx', sheet_name='ST20'))
state_emissions = state_emissions.drop(index = 0)
state_emissions_rt = state_emissions.iloc[:,[1,24]].copy()
state_emissions_rt = state_emissions_rt.rename(columns={'state_annual_co_2_equivalent_total_output_emission_rate_lb_m_wh':'co2e_factor'})
state_emissions_rt['co2e_factor'] = state_emissions_rt['co2e_factor'].astype(float)
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
g= sns.barplot(x=state_emissions_rt.iloc[:,0], y=state_emissions_rt.iloc[:,1],color ='gray')
plt.ylabel('Carbon Intensity (lbCO2e / mWh)')
plt.xlabel('State')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
y = pd.DataFrame(state_emissions_rt.columns,state_emissions_rt.dtypes)

subregion_emissions = clean_columns(pd.read_excel(r'egrid2020_data.xlsx', sheet_name='SRL20')).drop(index=0)
subregion_emissions_rt = subregion_emissions.iloc[:,[2,24]].copy()
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
g_sr= sns.barplot(x=subregion_emissions_rt.iloc[:,0], y=subregion_emissions_rt.iloc[:,1],color='gray')
plt.setp(g_sr.get_xticklabels(), rotation=90)
plt.xlabel('eGrid Subregion')
plt.ylabel('Carbon Intensity (lbCO2e / mWh)')
plt.show()

abs(532.979-1491.35)/1491.35

y = pd.DataFrame(subregion_emissions.columns)

# The mWh associated with one workload batch, num_transactions = batch
num_trans = 1000000
four_sockets_e = (four_sockets_scl.iloc[:,22:32]**-1)*num_trans #Run Time in Seconds
four_sockets_e = four_sockets_e.mul(four_sockets_scl['total_joules'],axis=0) #Energy Watts*Seconds
four_sockets_e = (four_sockets_e/3600)/1000000 #Conversion to MWh
def co2e_calc(st_abb):
    co2e_matrix = four_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']==st_abb,'co2e_factor'].iloc[0]
    return co2e_matrix

KY_emissions = co2e_calc('KY')

four_sockets_CAe = co2e_calc('CA')
four_sockets_TXe = co2e_calc('TX')

for col in subset_columns:
    sns.set_context("poster")
    plt.rcParams["figure.figsize"] = (24, 16)
    data = pd.DataFrame({'index':four_sockets_e.index,'CA':four_sockets_CAe[col], 'TX':four_sockets_TXe[col]})
    title = col.split('_eff')[0]
    title = title.upper()
    title = title.replace('_',' ')
    sns.barplot(data=data.melt(id_vars='index',value_name='Emissions (lbCO2e)', var_name='Data Center Location'), x='index',y='Emissions (lbCO2e)',hue='Data Center Location',palette='Greens').set(title=title)
    plt.show()

CAe_avg = four_sockets_CAe.iloc[:,0].mean()
TXe_avg = four_sockets_TXe.iloc[:,0].mean()
CAe_avg
TXe_avg


df4 = pd.DataFrame(four_sockets_CAe[subset_columns[0]])
df5 = pd.DataFrame(four_sockets_TXe[subset_columns[0]])
four_sockets['CA Compress Emissions'] = df4.copy()
four_sockets['TX Compress Emissions'] = df5.copy()
sns.set_color_codes("colorblind")
sns.barplot(x=four_sockets.index,y=four_sockets['CA Compress Emissions'])

# The Emissions associated with Core Count vs server index
four_sockets_core = four_sockets_scl.iloc[:,22:33].groupby(four_sockets_scl.iloc[:,9]).mean()
four_sockets_core['Average Watts'] = four_sockets_scl['total_joules'].groupby(four_sockets_scl.iloc[:,9]).mean()
four_sockets_core['CPU Average'] = four_sockets_core.iloc[:,[0,1,2,3,4,5,10]].mean(axis=1)
four_sockets_core['Memory Average'] = four_sockets_core.iloc[:,[6,7]].mean(axis=1)
four_sockets_core['Storage Average'] = four_sockets_core.iloc[:,[8,9]].mean(axis=1)
four_sockets_core = four_sockets_core.drop(four_sockets_core.columns[0:11],axis=1)
four_sockets_core['index'] = four_sockets_core.index
#four_sockets_core.iloc[:,1:4] = four_sockets_core.iloc[:,1:4].mul(four_sockets_core['Average Watts'],axis=0)
four_sockets_core_perf = pd.melt(four_sockets_core,id_vars = 'index', value_vars = four_sockets_core.iloc[:,1:4])
sns.barplot(x='index',y='value',hue='variable',data = four_sockets_core_perf)
plt.xlabel('Core Count', fontsize=20)
plt.ylabel('Transactions per Second', fontsize=20)
plt.legend(title = 'SERT Workload Category')

sns.scatterplot(data = four_sockets_core_perf, x='index', y ='value', hue = 'variable' )
plt.show()

def arch_emissions(st_abb,df):
    arch_e = df.copy()
    arch_e.iloc[:,1:4] = (arch_e.iloc[:,1:4]**-1)*1000000
    arch_e.iloc[:,1:4] = arch_e.iloc[:,1:4].mul(arch_e.iloc[:,0],axis=0)
    arch_e.iloc[:,1:4] = (arch_e.iloc[:,1:4]/3600)/1000000
    arch_e.iloc[:,1:4] = arch_e.iloc[:,1:4]*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']==st_abb,'co2e_factor'].iloc[0]
    arch_e = pd.melt(arch_e,id_vars = 'index', value_vars = arch_e.iloc[:,1:4])
    return arch_e

def arch_emissions2(st_abb,df):
    arch_e = df.copy()
    arch_e.iloc[:,1:4] = (arch_e.iloc[:,1:4]**-1)*1000000
    arch_e.iloc[:,1:4] = arch_e.iloc[:,1:4].mul(arch_e.iloc[:,0],axis=0)
    arch_e.iloc[:,1:4] = (arch_e.iloc[:,1:4]/3600)/1000000
    arch_e.iloc[:,1:4] = arch_e.iloc[:,1:4]*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']==st_abb,'co2e_factor'].iloc[0]
    return arch_e

# Averaging Emissions for each workload
CA_core_e = arch_emissions('CA',four_sockets_core)
sns.barplot(x='index',y='value',hue='variable',data = CA_core_e).set(title ='CA Average Emissions by Core Count')
plt.xlabel('Core Count', fontsize=20)
plt.ylabel('Emissions (lbCO2e)', fontsize=20)
plt.legend(title = 'SERT Workload Category')

# Regression for Core Count
core_reg = arch_emissions2('CA',four_sockets_core)
reg_cpu = np.poly1d(np.polyfit(core_reg['index'],core_reg['CPU Average'],1))
reg_sto = np.poly1d(np.polyfit(core_reg['index'],core_reg['Memory Average'],1))
reg_mem = np.poly1d(np.polyfit(core_reg['index'],core_reg['Storage Average'],1))
reg_line = np.linspace(np.amin(core_reg['index']),np.amax(core_reg['index']),9)

core_reg1 = core_reg.copy()
core_reg1 = core_reg1.drop(4)
x = core_reg['index']
x = sm.add_constant(x)
cpu_lin = sm.OLS(core_reg['CPU Average'],x)
cpu_res = cpu_lin.fit()
print(cpu_res.summary())
sto_lin = sm.OLS(core_reg['Storage Average'],x)
sto_res = sto_lin.fit()
print(sto_res.summary())
mem_lin = sm.OLS(core_reg['Memory Average'],x)
mem_res = mem_lin.fit()
print(mem_res.summary())

#Dropping First Point
core_reg1 = core_reg.copy()
core_reg1 = core_reg1.drop(4)
x1 = core_reg1['index']
x1 = sm.add_constant(x1)
cpu_lin1 = sm.OLS(core_reg1['CPU Average'],x1)
cpu_res1 = cpu_lin1.fit()
print(cpu_res1.summary())
sto_lin1 = sm.OLS(core_reg1['Storage Average'],x1)
sto_res1 = sto_lin1.fit()
print(sto_res1.summary())
mem_lin1 = sm.OLS(core_reg1['Memory Average'],x1)
mem_res1 = mem_lin1.fit()
print(mem_res1.summary())

core_reg1 = pd.melt(core_reg1,id_vars = 'index', value_vars = core_reg1.iloc[:,1:4])
plt.rcdefaults()
g = sns.scatterplot(data = core_reg1, x ='index',y='value',hue='variable',palette = ['blue','green','orange'])
plt.plot(x1,(0.0218)-0.0006*x1,color='blue')
plt.plot(x1,(0.051)-0.0023*x1,color='green')
plt.plot(x1,(0.0852)-0.0034*x1,color='orange')
plt.xlabel('Core Count')
plt.ylabel('Emissions (lbCO2e')
plt.legend(title = ' SERT Workload Category', labels=['CPU Average, R^2 = 0.374','Memory Average, R^2 = 0.449', 'Storage Average, R^2 = 0.452'])

# 4th Order Polynomial Model
x = core_reg['index']
core_reg2 = core_reg.copy()
core_reg2 = clean_columns(core_reg2)
cpu_model = 'cpu_average ~ index + I(index**2) +I(index**3)+I(index**4)'
cpu_pol = smf.ols(formula = cpu_model, data = core_reg2).fit()
print(cpu_pol.summary())
sto_model = 'storage_average ~ index + I(index**2) +I(index**3)+I(index**4)'
sto_pol = smf.ols(formula = sto_model, data = core_reg2).fit()
print(sto_pol.summary())
mem_model = 'memory_average ~ index + I(index**2) +I(index**3)+I(index**4)'
mem_pol = smf.ols(formula = mem_model, data = core_reg2).fit()
print(mem_pol.summary())


plt.rcdefaults()
g = sns.scatterplot(data = CA_core_e, x ='index',y='value',hue='variable',palette = ['blue','green','orange'])
plt.plot(x,(-0.058)+0.0263*x-0.0029*(x**2)+0.0001*(x**3)-(1.718*10**-6)*(x**4),color='blue')
plt.plot(x,(-0.194)+0.0788*x-0.0087*(x**2)+0.0004*(x**3)-(5.282*10**-6)*(x**4),color='green')
plt.plot(x,(-0.2641)+0.1073*x-0.0114*(x**2)+0.0005*(x**3)-(6.537*10**-6)*(x**4),color='orange')
plt.xlabel('Core Count')
plt.ylabel('Emissions (lbCO2e')
plt.legend(title = ' SERT Workload Category')

# 3rd Order Polynomial Model
x = core_reg['index']
core_reg2 = core_reg.copy()
core_reg2 = clean_columns(core_reg2)
cpu_model = 'cpu_average ~ index + I(index**2) +I(index**3)'
cpu_pol = smf.ols(formula = cpu_model, data = core_reg2).fit()
print(cpu_pol.summary())
sto_model = 'storage_average ~ index + I(index**2) +I(index**3)'
sto_pol = smf.ols(formula = sto_model, data = core_reg2).fit()
print(sto_pol.summary())
mem_model = 'memory_average ~ index + I(index**2) +I(index**3)'
mem_pol = smf.ols(formula = mem_model, data = core_reg2).fit()
print(mem_pol.summary())
plt.rcdefaults()
g = sns.scatterplot(data = CA_core_e, x ='index',y='value',hue='variable',palette = ['blue','green','orange'])
plt.plot(x,(-0.0066)+0.0062*x-0.0005*(x**2)+(1.047*10**-5)*(x**3),color='blue')
plt.plot(x,(-0.0358)+0.017*x-0.0013*(x**2)+(2.729*10**-5)*(x**3),color='green')
plt.plot(x,(-0.0683)+0.0308*x-0.0023*(x**2)+(4.719*10**-5)*(x**3),color='orange')
plt.xlabel('Core Count')
plt.ylabel('Emissions (lbCO2e')
plt.legend(title = ' SERT Workload Category')


(four_sockets_e.iloc[:,0].max()-four_sockets_e.iloc[:,0].min())/four_sockets_e.iloc[:,0].max()

four_sockets_CAe = four_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='CA','co2e_factor'].iloc[0]
four_sockets_TXe = four_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='TX','co2e_factor'].iloc[0]
four_sockets_VTe = four_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='VT','co2e_factor'].iloc[0]
four_sockets_WYe = four_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='WY','co2e_factor'].iloc[0]
best_VT = four_sockets_VTe.iloc[:,0].min() #61
wrst_WY = four_sockets_WYe.iloc[:,0].max() #191

wkld_avgs = pd.DataFrame(four_sockets_CAe.mean(axis = 0))
wkld_avgs['Workload'] = wkld_avgs.index.astype(str)
wkld_names = wkld_avgs['Workload'].str.split(pat = '_eff',n=1, expand = True)
wkld_avgs['Workload'] = wkld_names[0]
wkld_avgs['Workload'] = wkld_avgs['Workload'].apply(str.upper)
wkld_avgs = wkld_avgs.replace('_',' ',regex = True)
wkld_avgs = wkld_avgs.replace('SERT ','',regex = True)
wkld_avgs = wkld_avgs.sort_values(0)

plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(wkld_avgs['Workload'],wkld_avgs.iloc[:,0])
ax.set_xlabel('Emissions (lbsCO2e)', fontsize=10)
ax.set_ylabel('Workload', fontsize=10)
ax.set_title('Avg Emissions by Workload (CA)')
plt.show()
wkld_avgs.to_csv('wkldavgs.csv',index=False)





y = pd.DataFrame(four_sockets_e.columns,four_sockets_e.dtypes)



def mab_df(wkld,st_abb):
    em_mtx = co2e_calc(st_abb)
    df = pd.DataFrame(em_mtx[subset_columns[wkld]])
    colname = subset_columns[wkld].split('_eff')[0]
    df.rename(columns ={subset_columns[wkld]:st_abb+'_'+colname+'_emissions'},inplace = True)
    return df

df44 = mab_df(1,'CA')

class server:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def get_consumption_from_true_distribution(self):
        s = np.random.normal(self.mu, self.sigma)
        self.n += 1
        self.sum_consumption += s
        return s

class serverThompsonSampler(server):
    def __init__(self, mu, sigma):
        self.prior_mu_of_mu = 0
        self.prior_sigma_of_mu = 1000
        
        self.post_mu_of_mu = self.prior_mu_of_mu
        self.post_sigma_of_mu = self.prior_sigma_of_mu
        
        self.n = 0
        self.sum_consumption = 0
        
        super().__init__(mu, sigma)
        
    def get_mu_from_current_distribution(self):
        samp_mu = np.random.normal(self.post_mu_of_mu, self.post_sigma_of_mu)
        return samp_mu
    
    def update_current_distribution(self):
        self.post_sigma_of_mu = np.sqrt((1 / self.prior_sigma_of_mu**2 + self.n / self.sigma**2)**-1)
        self.post_mu_of_mu = (self.post_sigma_of_mu**2) * ((self.prior_mu_of_mu / self.prior_sigma_of_mu**2) + (self.sum_consumption / self.sigma**2))

def draw_distributions(R,i):
    for r in R:
        sns.set_context("poster")
        plt.rcParams["figure.figsize"] = (24, 16)
        samps = np.random.normal(r.post_mu_of_mu, r.post_sigma_of_mu, 10000)
        sns.kdeplot(samps, fill=True)
        plt.title('Iteration %s'%(i+1), fontsize=20)
        plt.legend(['mu=%s'%(r.mu) for r in R], fontsize=16)
        plt.xlim(-.05,.05)
        plt.xlabel('Workload Operational Emissions', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        #plt.savefig(f'./img/img_{i}.png', 
        #        transparent = False,  
        #        facecolor = 'white')
    plt.show()

def thomp_mab(dataframe,runs,population):
    num_servers = len(dataframe.index)
    R = [serverThompsonSampler(dataframe.iloc[i,0], 0.002) for i in range(0, num_servers)]
    frames = []
    for i in range(runs):
        if i < 10 or (i < 100 and (i+1) % 10 == 0) or ((i+1) % 100 == 0):
            draw_distributions(R,i)
            #image = imageio.v2.imread(f'./img/img_{i}.png')
            #frames.append(image)
           
        
        #get a sample from each posterior
        post_samps = [r.get_mu_from_current_distribution() for r in R]
        
        #index of distribution with highest consumption
        chosen_idx = post_samps.index(min(post_samps))
        
        #get a new sample from that distribution
        s = R[chosen_idx].get_consumption_from_true_distribution()
        
        #update that distributions posterior
        R[chosen_idx].update_current_distribution()
        optimal_server = pd.DataFrame(population.iloc[chosen_idx,:]).T
    return optimal_server#,frames


mab_dataset = pd.DataFrame({'CA Compress Emissions':four_sockets['CA Compress Emissions']})
CA_opt_server = thomp_mab(mab_dataset, 1000,four_sockets)
title = 'The Best Sever is '+str(CA_opt_server.index[0])+ ' in '+ CA_opt_server.iloc[0,1]
sns.barplot(x=mab_dataset.index, y='CA Compress Emissions',data = mab_dataset,color='Grey').set(title = title)
#imageio.mimsave('./CA_MAB.gif', CA_opt_server[1],fps = 1, loop=1)

# plt.figure(figsize=(5,5))
# true_means = [r.mu for r in R]
# posterior_means = [r.post_mu_of_mu for r in R]
# plt.scatter(true_means, posterior_means)
# plt.plot(true_means, true_means, color='k', alpha=0.5, linestyle='--')

# plt.xlabel('True Mean', fontsize=20)
# plt.ylabel('Posterior Mean', fontsize=20)
        

# Start the Random Mix of Hardware in the CA and TX datacenter split 50/50

rand_CA = four_sockets.sample(n=int(len(four_sockets.index)/2))
rand_CA = pd.DataFrame({'Emissions':rand_CA.iloc[:,41],'loc':'CA'})
rand_TX = four_sockets.drop(rand_CA.index)
rand_TX = pd.DataFrame({'Emissions':rand_TX.iloc[:,42],'loc':'TX'})
rand_port = pd.concat([rand_CA,rand_TX],axis=0,ignore_index = False)
rand_port = rand_port.sort_index()
rand_mab = thomp_mab(rand_port,1000,rand_port)
title = 'The Best Sever is '+str(rand_mab.index[0])+ ' in '+ rand_mab.iloc[0,1]
sns.barplot(x=rand_port.index, y='Emissions',hue = 'loc',data = rand_port,palette='Greys').set(title=title)


# Analyze 10 random servers in CA and TX

lim_CA = four_sockets.sample(n=10)
rmdr = four_sockets.drop(lim_CA.index)
lim_TX = rmdr.sample(n=10)
lim_CA = pd.DataFrame({'Emissions':lim_CA.iloc[:,41],'loc':'CA'})
lim_TX = pd.DataFrame({'Emissions':lim_TX.iloc[:,42],'loc':'TX'})
lim_port = pd.concat([lim_CA,lim_TX],axis=0,ignore_index=False)
lim_port = lim_port.sort_index()
lim_mab = thomp_mab(lim_port,1000,lim_port)
title = 'The Best Sever is '+str(lim_mab.index[0])+ ' in '+ lim_mab.iloc[0,1]
sns.barplot(x=lim_port.index, y='Emissions',hue = 'loc',data = lim_port,palette='Greys').set(title=title)

# Analyze 3 Different States

states = ['ID','WA','ME']

ID_compress_emissions = mab_df(0,'ID')
WA_compress_emissions = mab_df(0,'WA')
ME_compress_emissions = mab_df(0,'ME')
ID_compress_emissions = ID_compress_emissions.sample(n=10)
WA_compress_emissions = WA_compress_emissions.drop(ID_compress_emissions.index).sample(n=10)    
ME_compress_emissions = ME_compress_emissions.drop(ID_compress_emissions.index).drop(WA_compress_emissions.index)
ID_port = pd.DataFrame({'Emissions':ID_compress_emissions.iloc[:,0],'loc':'ID'})
WA_port = pd.DataFrame({'Emissions':WA_compress_emissions.iloc[:,0],'loc':'WA'})
ME_port = pd.DataFrame({'Emissions':ME_compress_emissions.iloc[:,0],'loc':'ME'})
tri_port = pd.concat([ID_port,WA_port,ME_port],axis=0,ignore_index=False)
tri_port = tri_port.sort_index()
tri_mab = thomp_mab(tri_port,1000,tri_port)
title = 'Server Emissions by Location for the Compress Workload'
sns.barplot(x=tri_port.index,y='Emissions',hue='loc',data=tri_port).set(title=title)
plt.legend(title = 'Location',loc = 'upper right')
tri_port2 = tri_port.copy()
tri_port2 = tri_port2['Emissions'].groupby(tri_port2['loc']).mean()
tri_port['Emissions'] = tri_port['Emissions'].sort_values(0)
