import pandas as pd
from skimpy import clean_columns
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


servers = pd.read_csv('certified-enterprise-servers-2023-01-12_clean.csv')
servers = clean_columns(servers)

two_socket_map = servers[servers['product_processor_socket_count']=='4']
two_socket_map = two_socket_map.iloc[:,[0,1]]
two_socket_map.to_csv('indexmap.csv',index=True)

pca = PCA(n_components=2)
two_sockets = servers[servers['product_processor_socket_count']=='2']
two_sockets = two_sockets.replace(' ','',regex = True)
two_sockets = two_sockets.replace('-','_',regex = True)
two_sockets = two_sockets.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
two_sockets.iloc[:,3:16] = two_sockets.iloc[:,3:16].astype(float)
two_sockets.iloc[:,18:95] = two_sockets.iloc[:,18:95].astype(float)
two_sockets.fillna(0,inplace = True)
two_sockets = two_sockets.applymap(lambda x: x.lower() if type(x) == str else x)
rowid = two_sockets.index
two_sockets_dum = pd.get_dummies(two_sockets, columns=['brand_name','form_factor','processor_brand_typical_or_single_configuration'] )
two_sockets_dum = two_sockets_dum.drop(two_sockets_dum.columns[0],axis=1)
two_socket_pca = pca.fit_transform(two_sockets_dum)
two_socket_pca = pd.DataFrame(two_socket_pca,index = rowid)
two_socket_pca = two_socket_pca.rename(columns = {0:'PCA1',1:'PCA2'})
two_sockets = two_sockets.join(two_socket_pca)
two_sockets['brand_name'] = two_sockets['brand_name'].astype(str)
two_sockets.loc[two_sockets['brand_name']=='dell','brand_name'] = 'dellemc'
two_sockets.loc[two_sockets['brand_name']=='hpeproliantdl560gen10','brand_name'] = 'hewlettpackardenterprise'
two_sockets.dtypes
two_sockets['total_joules']=two_sockets['power_supply_rated_output_typical_or_single_configuration_w']*two_sockets['num_power_supplies_for_typical_or_single_configuration']
two_sockets.fillna(0,inplace = True)
two_sockets = two_sockets.drop(two_sockets.columns[two_sockets.columns.str.contains('high')], axis=1)
two_sockets = two_sockets.drop(two_sockets.columns[two_sockets.columns.str.contains('low')], axis=1)
two_sockets.iloc[:,22:33] = two_sockets.iloc[:,22:33].apply(pd.to_numeric, errors='coerce', downcast='float')
two_sockets['total_joules'] = two_sockets['total_joules'].apply(pd.to_numeric, errors='coerce', downcast='float')
two_sockets_scl = two_sockets.copy()

########## Plotting Workload Efficiency Scores ##########################
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
df = two_sockets.copy()
subset_columns = df.columns[df.columns.get_indexer(df.columns.values[22:32])]
for col in subset_columns:
    sns.barplot(x=df.index, y=df[col],color='gray')
    plt.show()


two_sockets_scl.iloc[:,22:33] = two_sockets_scl.iloc[:,22:33].mul(two_sockets_scl['total_joules'], axis=0)


y = pd.DataFrame(two_sockets.columns)
plt.rcParams["figure.figsize"] = (12, 8)
sns.scatterplot(x ='PCA1', y ='PCA2',hue='brand_name',data = two_sockets)
plt.show()
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)

######## Server Capacity Graphs ########################################
df2 = two_sockets_scl.copy()
subset_columns = df2.columns[df2.columns.get_indexer(df2.columns.values[22:32])]
for col in subset_columns:
    sns.barplot(x=df2.index, y=df2[col],color='gray')
    plt.show()

################### Emissions Graphs ####################################
state_emissions = clean_columns(pd.read_excel(r'egrid2020_data.xlsx', sheet_name='ST20'))
state_emissions = state_emissions.drop(index = 0)
state_emissions_rt = state_emissions.iloc[:,[1,24]].copy()
state_emissions_rt = state_emissions_rt.rename(columns={'state_annual_co_2_equivalent_total_output_emission_rate_lb_m_wh':'co2e_factor'})
state_emissions_rt['co2e_factor'] = state_emissions_rt['co2e_factor'].astype(float)
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
g= sns.barplot(x=state_emissions_rt.iloc[:,0], y=state_emissions_rt.iloc[:,1],)
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
y = pd.DataFrame(state_emissions_rt.columns,state_emissions_rt.dtypes)

subregion_emissions = clean_columns(pd.read_excel(r'egrid2020_data.xlsx', sheet_name='SRL20')).drop(index=0)
subregion_emissions_rt = subregion_emissions.iloc[:,[2,24]].copy()
sns.set_context("poster")
plt.rcParams["figure.figsize"] = (24, 16)
g_sr= sns.barplot(x=subregion_emissions_rt.iloc[:,0], y=subregion_emissions_rt.iloc[:,1],color='gray')
plt.setp(g_sr.get_xticklabels(), rotation=90)
plt.show()

y = pd.DataFrame(subregion_emissions.columns)

# The emissions from one workload batch, num_transactions = batch
num_trans = 100000
two_sockets_e = (two_sockets.iloc[:,22:32]**-1)*num_trans #Run Time in Seconds
two_sockets_e = two_sockets_e.mul(two_sockets['total_joules'],axis=0) #Energy Watts*Seconds
two_sockets_e = (two_sockets_e/3600)/100000 #Conversion to MWh
def co2e_calc(st_abb):
    co2e_matrix = two_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']==st_abb,'co2e_factor'].iloc[0]
    return co2e_matrix

KY_emissions = co2e_calc('KY')

two_sockets_CAe = co2e_calc('CA')
two_sockets_TXe = co2e_calc('TX')

# two_sockets_CAe = two_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='CA','co2e_factor'].iloc[0]
# two_sockets_TXe = two_sockets_e*state_emissions_rt.loc[state_emissions_rt['state_abbreviation']=='TX','co2e_factor'].iloc[0]


for col in subset_columns:
    sns.set_context("poster")
    plt.rcParams["figure.figsize"] = (24, 16)
    data = pd.DataFrame({'index':two_sockets_e.index,'CA Emissions':two_sockets_CAe[col], 'TX Emissions':two_sockets_TXe[col]})
    sns.barplot(data=data.melt(id_vars='index',value_name='Emissions (lbCO2e)', var_name='Data Center Location'), x='index',y='Emissions (lbCO2e)',hue='Data Center Location',palette='Greys').set(title=col)
    plt.show()
    
df4 = pd.DataFrame(two_sockets_CAe[subset_columns[0]])
df5 = pd.DataFrame(two_sockets_TXe[subset_columns[0]])
two_sockets['CA Compress Emissions'] = df4.copy()
two_sockets['TX Compress Emissions'] = df5.copy()
sns.set_color_codes("colorblind")
sns.barplot(x=two_sockets.index,y=two_sockets['CA Compress Emissions'],color='gray')


y = pd.DataFrame(two_sockets_e.columns,two_sockets_e.dtypes)

mab_dataset = pd.DataFrame({'CA Compress Emissions':two_sockets['CA Compress Emissions']})

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
        samps = np.random.normal(r.post_mu_of_mu, r.post_sigma_of_mu, 10000)
        sns.kdeplot(samps, fill=True)
    plt.title('Iteration %s'%(i+1), fontsize=20)
    #plt.legend(['mu=%s'%(r.mu) for r in R], fontsize=16)
    plt.xlim(-200,200)
    plt.xlabel('SERT Power Consumption', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    
    plt.show()


def thomp_mab(dataframe,runs,population):
    num_servers = len(dataframe.index)
    R = [serverThompsonSampler(dataframe.iloc[i,0], 6) for i in range(0, num_servers)]
    for i in range(runs):
        if (i+1) % (runs/10) == 0:
            draw_distributions(R,i)
        
        #get a sample from each posterior
        post_samps = [r.get_mu_from_current_distribution() for r in R]
        
        #index of distribution with highest consumption
        chosen_idx = post_samps.index(min(post_samps))
        
        #get a new sample from that distribution
        s = R[chosen_idx].get_consumption_from_true_distribution()
        
        #update that distributions posterior
        R[chosen_idx].update_current_distribution()
        optimal_server = pd.DataFrame(population.iloc[chosen_idx,:]).T
    return optimal_server

CA_opt_server = thomp_mab(mab_dataset, 10000,two_sockets)

plt.figure(figsize=(5,5))
true_means = [r.mu for r in R]
posterior_means = [r.post_mu_of_mu for r in R]
plt.scatter(true_means, posterior_means)
plt.plot(true_means, true_means, color='k', alpha=0.5, linestyle='--')

plt.xlabel('True Mean', fontsize=20)
plt.ylabel('Posterior Mean', fontsize=20)
        

# Start the Random Mix of Hardware in the CA and TX datacenter split 50/50

rand_CA = two_sockets.sample(n=int(len(two_sockets.index)/2))
rand_CA = pd.DataFrame({'Emissions':rand_CA.iloc[:,41],'loc':'CA'})
rand_TX = two_sockets.drop(rand_CA.index)
rand_TX = pd.DataFrame({'Emissions':rand_TX.iloc[:,42],'loc':'TX'})
rand_port = pd.concat([rand_CA,rand_TX],axis=0,ignore_index = False)
rand_port = rand_port.sort_index()
rand_mab = thomp_mab(rand_port,10000,rand_port)
title = 'The Best Sever is '+str(rand_mab.index[0])+ ' in '+ rand_mab.iloc[0,1]
sns.barplot(x=rand_port.index, y='Emissions',hue = 'loc',data = rand_port,palette='Greys').set(title=title)

# Analyze 10 random servers in CA and TX

lim_CA = two_sockets.sample(n=10)
rmdr = two_sockets.drop(lim_CA.index)
lim_TX = rmdr.sample(n=10)
lim_CA = pd.DataFrame({'Emissions':lim_CA.iloc[:,41],'loc':'CA'})
lim_TX = pd.DataFrame({'Emissions':lim_TX.iloc[:,42],'loc':'TX'})
lim_port = pd.concat([lim_CA,lim_TX],axis=0,ignore_index=False)
lim_port = lim_port.sort_index()
lim_mab = thomp_mab(lim_port,1000,lim_port)
title = 'The Best Sever is '+str(lim_mab.index[0])+ ' in '+ lim_mab.iloc[0,1]
sns.barplot(x=lim_port.index, y='Emissions',hue = 'loc',data = lim_port,palette='Greys').set(title=title)
check = lim_mab.min(axis=0)
check
# Analyze 3 Different States

states = ['ID','WA','ME']

ID_compress_emissions = mab_df(0,'ID')
WA_compress_emissions = mab_df(0,'WA')
ME_compress_emissions = mab_df(0,'ME')
ID_compress_emissions = ID_compress_emissions.sample(n=int(len(ID_compress_emissions)/3))
WA_compress_emissions = WA_compress_emissions.drop(ID_compress_emissions.index).sample(n=10)    
ME_compress_emissions = ME_compress_emissions.drop(ID_compress_emissions.index).drop(WA_compress_emissions.index)
ID_port = pd.DataFrame({'Emissions':ID_compress_emissions.iloc[:,0],'loc':'ID'})
WA_port = pd.DataFrame({'Emissions':WA_compress_emissions.iloc[:,0],'loc':'WA'})
ME_port = pd.DataFrame({'Emissions':ME_compress_emissions.iloc[:,0],'loc':'ME'})
tri_port = pd.concat([ID_port,WA_port,ME_port],axis=0,ignore_index=False)
tri_port = tri_port.sort_index()
tri_mab = thomp_mab(tri_port,1000,tri_port)
title = 'The Best Sever is '+str(tri_mab.index[0])+ ' in '+ tri_mab.iloc[0,1]
sns.barplot(x=tri_port.index,y='Emissions',hue='loc',data=tri_port).set(title=title)
check = tri_mab.min(axis=0)
check
