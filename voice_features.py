import time
import os
import math
import pylab
import glob
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'font.serif': ['Computer Modern Roman'],
#     'font.size': 12,
#     'text.usetex': True,
#     'pgf.rcfonts': False,})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font='Computer Modern Roman', font_scale = 1.3)
import pingouin as pg
import scipy.stats
import scipy.stats as stats
from scipy import stats
from scipy.stats import norm
#from statsmodels.stats import shapiro

def vf_load(data_dir):
    ## Load CAMP data
    df = pd.read_csv(os.path.join(data_dir, "CAMP_study_data.csv"), sep =';', decimal=",", index_col='PID') ## Import dataframe
    df.index = df.index.astype(int)
    pat = sorted(df.index.unique()) # get list of PIDs (patient IDs)
    feat_names = df.columns.tolist() # get list of features
    ## Clean CAMP data
    df = df[df.voice != 0] # remove videos with no patient voice
    df = df[df["noise (human)"] == 0] # remove videos with noise from human
    # df = df[df.day != 0]  # remove instruction video at day 0, since it includes other voices #print(df.loc[df['day']==0].index.tolist()) --> see which PID have day=0
    df = df[df["noise (other)"] == 0] # remove videos with other noise

    df.reset_index(inplace=True)
    df.drop_duplicates(subset=['PID','rec_date'], keep='first',inplace=True)
    df.set_index('PID', inplace=True)

    f0_col = 18 # feature number according to excel table
    f0_col = f0_col - 2
    df.dropna(subset = [df.columns[f0_col]], inplace=True)
    for col in range(f0_col,f0_col+10):
        f_Hz(df, col)

    return df, pat

def f_Hz(df, feat): # conversion from semitone-frequency (octave) scale to normal frequency (Hz)
    f_0 = 27.5 # Hz
    df.iloc[:,feat] = f_0 * 2 ** ((df.iloc[:,feat])/12)
    #f = f_0 * 2 ** (f/12)

def norm_vf(df, col=None): # normalization
    col = df.columns
    for i in col:
        max = df['%s' % i].max() # calculate max of column 'colname'
        min = df['%s' % i].min() # calculate min of column 'colname'
        m = df['%s' % i].mean() # calculate mean of column 'colname'
        df[:]['%s' % i] = df[:]['%s' % i] - min
        df[:]['%s' % i] = df[:]['%s' % i] / (max - min)

def stand_vf(df, col):
    for i in col:
        m = df['%s' % i].mean() # calculate mean of column 'colname'
        sd = df['%s' % i].std() # calculate std of column 'colname'
        df[:]['%s' % i] = df[:]['%s' % i] - m
        if sd != 0:
            df[:]['%s' % i] = df[:]['%s' % i] / sd


if __name__ == '__main__':

    def data(df_vf):
        data = np.array(df_vf["day"])
        d = np.diff(np.unique(data)).min()
        left_of_first_bin = data.min() - float(d)/2
        right_of_last_bin = data.max() + float(d)/2

        plt.figure()
        plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d))
        #plt.title("Voice feature data per day")
        plt.xlabel("Day")
        plt.ylabel("Count")
        plt.grid(False)

        #sns.displot(data=data, kde=True)

        rootdir_images = os.path.expanduser(os.path.join("~", "polybox", "ETH_Master_Arbeit", "Images"))
        plt.savefig(os.path.join(rootdir_images,'videos.svg'))
        plt.savefig(os.path.join(rootdir_images,'videos.png'))
        plt.show()

    def demographics(df_vf): ## Demographics of PIDs
        rootdir = os.path.expanduser(os.path.join("~", "polybox", "ETH_Master_Arbeit", "CAMP"))
        df = pd.read_csv("%s/PID_info.csv" %rootdir)#, index_col='PID') ## Import dataframe
        #df = df[df['PID'].isin(df_vf.index.tolist())]

        df_male = df[df["sex"] == 1]
        df_male = df_male.sort_values(by=['age'])
        n_male = len(df_male.index.unique())
        #df_male.to_csv(os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "PID_rec_data_male.csv")))
        df_female = df[df["sex"] == 2]
        df_female = df_female.sort_values(by=['age'])
        n_female = len(df_female.index.unique())
        #df_female.to_csv(os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "PID_rec_data_female.csv")))
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
        # sex = ["male", "female"]
        n_s, bins_s, patches_s = ax1.hist(df_male["age"], [20, 30, 40, 50, 60, 70, 80, 90], histtype='bar') # .bar?
        n_a, bins_a, patches_a = ax2.hist(df_female["age"], [20, 30, 40, 50, 60, 70, 80, 90], histtype='bar') #[20, 40, 60, 80, 100])
        # n_a, bins_a, patches_a = ax2.hist(df["age"],[0, 20, 30, 40, 50, 60, 70, 80, 100])
        # n_a, bins_a, patches_a = ax2.hist(df["age"],[0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100])

        ## Separate subplots
        ax1.set_title(r'Male (n=%d)' % n_male)
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Count")
        #ax1.grid(True)
        ax1.set_ylim(0, 60)
        ax2.set_title("Female (n=%d)" %n_female)
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Count")
        ax2.set_ylim(0, 60)
        #ax2.grid(True)
        plt.tight_layout()
        #plt.show()

        ## One plot
        plt.figure(figsize=(6, 4))
        # sex = ["male", "female"]
        x = [df_male["age"], df_female["age"]]
        plt.hist(x, [20, 30, 40, 50, 60, 70, 80, 90], histtype='bar')
        # n_s, bins_s, patches_s = ax1.hist(df_male["age"], [20, 30, 40, 50, 60, 70, 80, 90], histtype='bar') # .bar?
        # n_a, bins_a, patches_a = ax2.hist(df_female["age"], [20, 30, 40, 50, 60, 70, 80, 90], histtype='bar') #[20, 40, 60, 80, 100])
        # n_a, bins_a, patches_a = ax2.hist(df["age"],[0, 20, 30, 40, 50, 60, 70, 80, 100])
        # n_a, bins_a, patches_a = ax2.hist(df["age"],[0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100])

        #plt.legend(['male','female'])
        #plt.legend([r'Male (n=%d)' %n_male,'Female (n=%d)' %n_female], loc='best')
        plt.legend([r'Male','Female'], loc='best')
        #plt.title(r'Male (n=%d), Female (n=%d)' % (n_male,n_female))
        plt.xlabel("Age")
        plt.ylabel("Count")
        #ax.grid(True)
        plt.ylim(0, None)
        plt.tight_layout()
        plt.grid(False)

        rootdir_images = os.path.expanduser(os.path.join("~", "polybox", "ETH_Master_Arbeit", "Images"))
        plt.savefig(os.path.join(rootdir_images,'participants.svg'))
        plt.savefig(os.path.join(rootdir_images,'participants.png'))
        #sns.catplot(data=df, kind="bar", x="age", y= hue="sex")
        plt.show()

        print(df_male["age"].agg(["count", "min", "max", "median", "mean", "std"]))
        print(df_female["age"].agg(["count", "min", "max", "median", "mean", "std"]))


    ## Dependent t-test for vf means
    def stats_ttest_dep(df_vf, df_PANAS, df_demo): # dependent t-test
        # H_0 (Null Hypothesis): mean_1 == mean_2
        alpha = 0.05 # if p < alpha, reject; if p > alpha --> fail to reject.
        # degrees of freedom df = number of scores - 1 = number of patients - 1
        cv = 1.98 # --> critical value = 1.98 --> if t > 1.98 or t < -1.98, then reject H_0 --> there is a significant difference in means with 95% confidence
        # df.dropna(inplace=True)
        list_dep = []
        def plot(col1,col2):
                        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(8, 5))
                        ax1.hist(col1)
                        ax1.set_title("Voice features Start")
                        ax1.set_xlabel("Values")
                        ax1.set_ylabel("Frequency")

                        ax2.hist(col2)
                        ax2.set_title("Voice features End")
                        ax2.set_xlabel("Values")
                        ax2.set_ylabel("Frequency")
                        plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))
                        plt.tight_layout()
                        plt.show()
        for sex in range(1,3):# = 1 # 1=male, 2=female
            for age_min in range(40,70,40):
                if age_min >= 40:
                    age_max = age_min + 19 + 11
                else:
                    age_max = age_min + 39
                age_min = df_demo['age'].min()
                age_max = df_demo['age'].max()
                df_gender = df_demo[(df_demo['sex'] == sex)]
                df_age = df_gender[(df_demo['age'] >= age_min) & (df_demo['age'] <= age_max)]
                ## Dependent t-test according to patient data in chosen days
                pv = df_panas_vf(df_PANAS,df_vf,pat_not,df_age, norm="no")
                #mv = df_mnh_vf(df_MNH,df_vf,pat_not,df_age)

                # mv_group = mv[mv['PID'].isin(df_age['PID'])]
                # mv_group_1 = mv[mv['date']=='2']
                # mv_group_2 = mv[mv['date']=='B']
                # pv_group = pv[pv['PID'].isin(df_age['PID'])]
                pv_group_1 = pv[pv['date']=='A']
                pv_group_2 = pv[pv['date']=='B']
                for f in pv.columns[6:]:
                    print("Sex %d, age %d - %d, feature %s" %(sex, age_min, age_max, f))

                    # H_0 (Null Hypothesis): mean_1 == mean_2
                    alpha = 0.05
                    col1 = pv_group_1[f]
                    col2 = pv_group_2[f]
                    col1.dropna(inplace=True)
                    col2.dropna(inplace=True)

                    # Scipy normal
                    SC = scipy.stats.ttest_rel(col1, col2, nan_policy="omit")
                    p_sc = SC[1]

                    # Scipy Wilcoxon
                    SC_W = scipy.stats.wilcoxon(col1, col2)
                    p_sc_w = SC_W[1]

                    # Pingouin normal
                    PIN = pg.ttest(col1, col2, paired=True)
                    p_pingouin = PIN.iloc[0,3]

                    # Pingouin Wilcoxon
                    PIN_W = pg.wilcoxon(col1, col2)
                    p_pingouin_w = abs(PIN_W.iloc[0,2])
                    # # Manual
                    # col1 = c1.reset_index(drop=True)
                    # col2 = c2.reset_index(drop=True)
                    # diff = col1 - col2 # calculate difference between scores for each participant
                    # diff.dropna(inplace=True)
                    #
                    # mean_col1 = col1.mean()
                    # se_col1 = col1.std() / len(col1)
                    # mean_col2 = col2.mean()
                    # se_col2 = col2.std() / len(col1)
                    #
                    # N = len(diff)
                    # dof = N - 1
                    # mean_diff = diff.mean()
                    # sd_diff = diff.std()
                    # T = mean_diff / (sd_diff/math.sqrt(N))
                    # r = math.sqrt((T**2)/(T**2 + N-1))
                    ## r = .1 small effect (explains 1% of total variance)
                    ## r = .3 medium effect (accounts for 9% of total variance)
                    ## r = .5 large effect (accounts for 25% of variance)
                    result = False
                    if p_sc < alpha or p_sc_w < alpha or p_pingouin < alpha or p_pingouin_w < alpha:
                        result = True

                    print("Pingouin:", p_pingouin, "Scipy:", p_sc)
                    print(SC)
                    print(PIN.iloc[0,0:3])
                    print("Wilcoxon")
                    print("Pingouin:", p_pingouin_w, "Scipy:", p_sc_w)
                    print(SC_W)
                    print(PIN_W.iloc[0,0:3])

                    if result:
                        print("H_0 rejected, there is a significant difference in means")
                        #plot(col1,col2)
                        print("M1 =", round(col1.mean(),2), ", SE1 =", round(col1.std(),2), "and M2 =", round(col2.mean(),2), "SE2 =", round(col2.std(), 2))
                        print('')
                        print('')
                        list_dep.append([sex, age_min, f])
                    else:
                        print("H_0 not rejected, no significant difference in means.")
                        print('')
                        print('')
        print(list_dep)

    ## Independent t-test for vf means
    def stats_ttest_indep(df_vf, df_demo): # independent t-test
        list_indep = []
        def plot(col1,col2):
                        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(8, 5))
                        ax1.hist(col1)
                        ax1.set_title("Voice features Start")
                        ax1.set_xlabel("Values")
                        ax1.set_ylabel("Frequency")

                        ax2.hist(col2)
                        ax2.set_title("Voice features End")
                        ax2.set_xlabel("Values")
                        ax2.set_ylabel("Frequency")
                        plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))
                        plt.tight_layout()
                        plt.show()
        for sex in range(1,3):# = 1 # 1=male, 2=female
            for age_min in range(40,70,40):
                if age_min >= 40:
                    age_max = age_min + 19 + 11
                else:
                    age_max = age_min + 39
                age_min = df_demo['age'].min()
                age_max = df_demo['age'].max()
                df_gender = df_demo[(df_demo['sex'] == sex)]
                df_age = df_gender[(df_demo['age'] >= age_min) & (df_demo['age'] <= age_max)]

                ## Independent t-test according to frequency of voice data per day
                df = df_vf[df_vf.index.isin(df_age['PID'])]
                data(df)
                df1 = df[df["day"]==2]
                df2 = df[df["day"]==28]
                for f in df.columns[16:]:
                    print("Sex %d, age %d - %d, feature %s" %(sex, age_min, age_max, f))
                    # H_0 (Null Hypothesis): mean_1 == mean_2
                    alpha = 0.05
                    # degrees of freedom df = number of scores - 1 = number of patients - 1
                    col1 = df1[f]
                    col2 = df2[f]
                    # col1.dropna(inplace=True)
                    # col2.dropna(inplace=True)

                    # Scipy
                    SC = scipy.stats.ttest_ind(col1, col2, equal_var=True, nan_policy="omit")
                    p_scipy = SC[1]

                    # Pingouin
                    PIN = pg.ttest(col1, col2, paired=False, correction=True)
                    p_pingouin = PIN.iloc[0,3]

                    result = False
                    if p_scipy <= alpha or p_pingouin <= alpha:
                        result = True

                    print("Pingouin:", p_pingouin, "Scipy:", p_scipy )
                    print(SC)
                    print(PIN.iloc[0,0:3])

                    if result:
                        print("H_0 rejected, there is a significant difference in means")
                        print('')
                        print('')
                        #plot(col1,col2)
                        list_indep.append([sex, age_min, f])
                    else:
                        print("H_0 not rejected")
                        print('')
                        print('')
        print(list_indep)

    start = time.time()

    df_vf, pat = vf_load()
    demo_dir = os.path.expanduser(os.path.join("~", "polybox", "ETH_Master_Arbeit", "CAMP"))
    df_demo = pd.read_csv("%s/PID_info.csv" %demo_dir)#, index_col='PID') ## Import dataframe
    t = ['A', 'B']#, 'C', 'D']

    #df_vf = df_vf.loc[df_vf['day'] == 2]
    #data(df_vf)
    demographics(df_vf)

    end = time.time()
    print((end-start)/60)
