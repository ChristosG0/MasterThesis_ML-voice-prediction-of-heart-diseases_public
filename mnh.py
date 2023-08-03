import time
import os
import numpy as np
import pingouin as pg
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import glob
from scipy import stats
from scipy.stats import norm
#from statsmodels.stats import shapiro

from voice_features import vf_load, norm_vf, stand_vf

## Loading and adjusting MNH
def MNH_load(data_dir): ### MNH (MacNewHeart)
    rootdir = os.path.join(data_dir, "questionnaires", "MNH")
    mnh1 = pd.read_csv("%s/MNH_1.csv" %rootdir, index_col=0)
    mnh2 = pd.read_csv("%s/MNH_2.csv" %rootdir, index_col=0)
    mnhB = pd.read_csv("%s/MNH_B.csv" %rootdir, index_col=0)
    mnhC = pd.read_csv("%s/MNH_C.csv" %rootdir, index_col=0)
    mnhD = pd.read_csv("%s/MNH_D.csv" %rootdir, index_col=0)
    mnh_X = [mnh1, mnh2, mnhB, mnhC, mnhD]
    mnhD[mnhD.columns[8]] = mnhC[mnhC.columns[8]]

    for m in mnh_X:
        m.replace('#NULL!',np.NaN, inplace=True) # replace empty cells / cells with space with NaN
        m[m.columns[0:6]] = m[m.columns[0:6]].apply(pd.to_numeric)
        for l in range(6,9):
            m[m.columns[l]] = pd.to_datetime(m[m.columns[l]], dayfirst = False)
        #p[p.columns[8]] = p[p.columns[8]].dt.date
        m[m.columns[9:]] = m[m.columns[9:]].apply(pd.to_numeric)

    for i in range(len(mnhC)):
        if pd.isnull(mnhB.iloc[i,8]): # if C has no datestamp, copy from B
            mnhB.iloc[i,8] = mnh2.iloc[i,8] + pd.DateOffset(months=1)
        mnhC.iloc[i,8] = mnhB.iloc[i,8] + pd.DateOffset(weeks=1)
    mnhD[mnhD.columns[8]] = mnhC[mnhC.columns[8]] + pd.DateOffset(weeks=1) # if D has no datestamp, copy from B

    df = mnh1.iloc[:,:8] # pd.DataFrame(columns=["PID"]) #, "ADate" ])
    for j in range(len(mnh_X)): # put all MNH together
        df = df.join(mnh_X[j].iloc[:,8])
        df = df.join(mnh_X[j].iloc[:,-1])
    df.index = df.index.astype(int)
    return df, mnh_X

def norm_q(df, col): # normalization
    for i in col:
        max = df['%s' % i].max() # calculate max of column 'colname'
        min = df['%s' % i].min() # calculate min of column 'colname'
        m = df['%s' % i].mean() # calculate mean of column 'colname'
        df[:]['%s' % i] = df[:]['%s' % i] - min
        df[:]['%s' % i] = df[:]['%s' % i] / (max - min)


def stand_q(df, col):
    for i in col:
        m = df['%s' % i].mean() # calculate mean of column 'colname'
        sd = df['%s' % i].std() # calculate std of column 'colname'
        df[:]['%s' % i] = df[:]['%s' % i] - m
        if sd != 0:
            df[:]['%s' % i] = df[:]['%s' % i] / sd

def df_mnh_vf(df_MNH,df_vf,df_age): # Correlation of MNH with voice -- tbf
    # Dataframe of MNH
    mnh = df_MNH.loc[df_MNH.index.isin(df_age['PID'])]
    mnh = mnh[mnh.index.isin(df_vf.index.unique())]
    # which_pat = "all"
    # which_pat = "pat decr"
    # mnh = mnh[~mnh.index.isin(pat_not)]
    mnh.drop(mnh.columns[list(range(4))+ [6,7] + list(range(14,18))], axis = 1, inplace = True)
    mnh.reset_index(drop=False, inplace=True)
    mnh_1 = mnh.iloc[:,:5]
    mnh_2 = mnh.iloc[:,:7]
    mnh_2.drop(mnh_2.columns[3:5], axis=1, inplace=True)
    mnh_B = mnh.iloc[:,:9]
    mnh_B.drop(mnh_B.columns[3:7], axis=1, inplace=True)
    for mnh_x in [mnh_1, mnh_2, mnh_B]:
        mnh_x = mnh_x.set_axis(['PID', 'sex', 'age', 'date', 'mnh'], axis=1, inplace=True)
    mnh_1.loc[:,'date'] = '1'
    mnh_2.loc[:,'date'] = '2'
    mnh_B.loc[:,'date'] = 'B'
    m = pd.concat([mnh_1, mnh_2, mnh_B])
    m.sort_values(by=['PID', 'date'], inplace=True)

    # Dataframe of Voice features
    vf = df_vf[df_vf.index.isin(mnh['PID'])]
    # vf = vf[~vf.index.isin(pat_not_es)]
    vf.loc[:,'rec_date'] = pd.to_datetime(vf.loc[:,'rec_date'], dayfirst = True)
    vf_1 = pd.DataFrame(columns=vf.columns)
    vf_2 = pd.DataFrame(columns=vf.columns)
    vf_B = pd.DataFrame(columns=vf.columns)
    for vf_x in [vf_1, vf_2, vf_B]:
        vf_x.drop(['rec_date'], axis=1, inplace=True)
    i=0
    for pid in list(vf.index.unique()):
        mnh_pid = mnh[mnh['PID'] == pid]
        d1 = mnh_pid.iloc[0,3]
        d2 = mnh_pid.iloc[0,5]
        dB = mnh_pid.iloc[0,7]
        vf_pid = vf.loc[vf.index == pid]
        vf_1_pid = vf_pid[vf_pid['rec_date'] == d1]
        vf_2_pid = vf_pid[vf_pid['rec_date'] == d2]
        vf_B_pid = vf_pid[vf_pid['rec_date'] == dB]
        vf_pid.sort_values(by=['day', 'minute'])
        vf_1 = pd.concat([vf_1, vf_1_pid])
        vf_2 = pd.concat([vf_2, vf_2_pid])
        vf_B = pd.concat([vf_B, vf_B_pid])

    for vf_x in [vf_1, vf_2, vf_B]:
        vf_x.reset_index(drop=False, inplace=True)
        vf_x.drop(vf_x.columns[list(range(1,15))], axis = 1, inplace = True)
        vf_x.rename(columns={"index": "PID", "mean_AP": "date"}, inplace = True)
    vf_1.loc[:,'date'] = '1'
    vf_2.loc[:,'date'] = '2'
    vf_B.loc[:,'date'] = 'B'
    v = pd.DataFrame(columns=vf_1.columns)
    v = pd.concat([vf_1, vf_2, vf_B], axis=0)

    mv = m.merge(v, how='inner', on=['PID','date'])
    # mv_group = mv.groupby("date").mean()
    norm_q(mv, mv.columns[4:-1].tolist())
    return mv

if __name__ == '__main__':

    def MNH_plots(pat, df, mnh_X): ## Plot MNH mean of all patients and MNH of each patient
        #fig = plt.figure(figsize=(12,5))
        m = ['mnh1', 'mnh2', 'mnhB', 'mnhC', 'mnhD']
        # between subjects
        for j in range(len(mnh_X)):
            c = 8+2*j
            # plt.axvline(x=df.iloc[0,6], color='grey') #, label="Entrance")
            # plt.text(df.iloc[0,6],0,'Entrance',rotation=80)
            # plt.axvline(x=df.iloc[0,7], color='grey') #, label="Entrance")
            # plt.text(df.iloc[0,7],0,'Exit',rotation=80)
            for k in range(len(df.index)):
                plt.scatter(m[j], df.iloc[k,c+1])
        plt.title('Mac New Heart')
        plt.legend(loc='best')
        plt.xlabel("Date")
        plt.ylabel("MNH Score")
        plt.gcf().autofmt_xdate()

        plt.show()
        #fig.savefig(os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "plots", "shimmer" + '_%s.png'%pat[i])))

        # within subjects
        for i in range(len(pat)): # iterate through all PIDs (## within subjects)
            m = df.loc[df.index == pat[i]] # create dataframe for each PID
            mnh_sum = [m.iloc[0,9+2*x] for x in range(5)]
            # for x in range(5): # normalize feature
            #     c = 9+2*x
            #     m.iloc[0,c] = norm_q(m.iloc[0,c],mnh_sum)
            #fig = plt.figure(figsize=(12,5))
            for j in range(len(mnh_X)):
                if not m.empty:# and m.iloc[0,j] != 0 and not pd.isna(m.iloc[0,j]):
                    c = 8+2*j
                    plt.axvline(x=m.iloc[:,6], color='grey') #, label="Entrance")
                    plt.text(m.iloc[:,6],0,'Entrance',rotation=80)
                    plt.axvline(x=m.iloc[:,7], color='grey') #, label="Entrance")
                    plt.text(m.iloc[:,7],0,'Exit',rotation=80)
                    plt.scatter(m.iloc[:,c], m.iloc[:,c+1], label = m.columns[c+1], marker='x', color='black')
            plt.title(pat[i])
            plt.legend(loc='best')
            plt.xlabel("Date")
            plt.ylabel("MNH Score")
            plt.gcf().autofmt_xdate()

            plt.show()
            #fig.savefig(os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "plots", "shimmer" + '_%s.png'%pat[i])))

    def MNH_stats_basic(df): ## Histogram and distribution of MNH
        df.dropna(inplace=True)
        df.drop(columns=df.columns[14:19], axis=1, inplace=True)
        for i in range(3):
            plt.figure()
            l = 9+2*i
            h = df.columns[l]
            m = df[h].mean()
            print(m)
            sd = df[h].std()
            plt.hist(df[h])
            plt.title('Histogram of %s' %h)
            #plt.legend(loc='best')
            plt.xlabel(h)
            plt.ylabel("Frequency")
            plt.text(3*m/4, m/4, 'mean=%d' %m)
            plt.grid(True)

            plt.figure()
            stats.probplot(df[h], dist="norm", plot=pylab)
            #plt.show()

            plt.figure()
            df[h].plot(kind='box')
            plt.show()

            df[h] = norm.rvs(size=len(df[h]))
            print(stats.shapiro(df[h]))

            rng = np.random.default_rng()
            df[h] = stats.norm.rvs(loc=5, scale=3, size=len(df), random_state=rng)
            shapiro_test = stats.shapiro(df[h])
            print(shapiro_test)


    def normality(df): ## Test normality of MNH data
        time = ['1', '2', 'B', 'C', 'D']
        m = 'shapiro'
        #m = 'normaltest'
        for t in time:
            print(pg.normality(df['mnh_%s_sum' %t], method=m, alpha = 0.05))

    def MNH_stats_ttest_dep(df): # Dependent t-test for MNH means
        # df.drop(columns=df.columns[14:19], axis=1, inplace=True)
        # H_0 (Null Hypothesis): mean_1 == mean_2
        # alpha = 0.05
        # degrees of freedom df = number of scores - 1 = number of patients - 1
        cv = 1.98 # --> critical value = 1.98 --> if t > 1.98 or t < -1.98, then reject H_0 --> there is a significant difference in means with 95% confidence
        # df.dropna(inplace=True)

        col1 = df['mnh_2_sum']
        col2 = df['mnh_B_sum']
        col1.dropna(inplace=True)
        col2.dropna(inplace=True)
        diff = col1 - col2 # calculate difference between scores for each participant
        diff.dropna(inplace=True)

        mean_col1 = col1.mean()
        se_col1 = col1.std() / len(col1)
        mean_col2 = col2.mean()
        se_col2 = col2.std() / len(col1)

        N = len(diff)
        dof = N - 1
        mean_diff = diff.mean()
        sd_diff = diff.std()
        t = mean_diff / (sd_diff/math.sqrt(N))
        r = math.sqrt((t**2)/(t**2 + N-1))
        ## r = .1 small effect (explains 1% of total variance)
        ## r = .3 medium effect (accounts for 9% of total variance)
        ## r = .5 large effect (accounts for 25% of variance)
        result = abs(t) > cv
        if result:
            print("H_0 rejected, there is a significant difference in means.")
            print("M1 =", round(mean_col1,2), ", SE1 =", round(se_col1,2), "and M2 =", round(mean_col2,2), "SE2 =", round(se_col2, 2))
            print("t(dof=%d) ="%dof, round(abs(t),2),">",cv, "for p <", 0.05, ", Effect size:", round(r,2))
        else:
            print("H_0 not rejected")
            print(abs(t), "<" ,cv,result,"dof =", N-1)

    def MNH_stats_ttest_indep(df): # Independent t-test for MNH means
        #df.drop(columns=df.columns[14:19], axis=1, inplace=True)
        # H_0 (Null Hypothesis): mean_1 == mean_2
        # alpha = 0.05
        # degrees of freedom df = number of scores - 1 = number of patients - 1
        col1 = df['mnh_1_sum']
        col2 = df['mnh_2_sum']
        col1.dropna(inplace=True)
        col2.dropna(inplace=True)

        n1 = len(col1)
        n2 = len(col2)
        deg1 = n1 - 1
        deg2 = n2 - 1
        deg = deg1 + deg2
        cv = 1.96 # = critical value --> if |t| > cv , then reject H_0 --> there is a significant difference in means with 95% confidence

        mean_col1 = col1.mean()
        mean_col2 = col2.mean()
        sd1 = col1.std()
        sd2 = col2.std()
        ss1 = deg1 * sd1**2
        ss2 = deg2 * sd2**2
        s_e = (ss1 + ss2) / deg

        t = (mean_col1 - mean_col2) / math.sqrt((s_e / n1) + (s_e / n2))
        #t = (mean_col1 - mean_col2) / math.sqrt((sd1**2 / n1) + (sd2**2 / n2))
        result = abs(t) > cv
        if result:
            print("H_0 rejected, there is a significant difference in means")
            print(abs(t), ">" ,cv,result,"dof =", deg)
        else:
            print("H_0 not rejected")
            print(abs(t), ">" ,cv,result,"dof =", deg)

        # plt.figure()
        # plt.hist(col1)
        # plt.figure()
        # plt.hist(col2)
        # plt.show()

    ## Correlation of Voice Features with MNH
    def correl_mnh_plot_all(mv,t, R, f, sex, age_min, age_max): # Function for Correlation Matrix function: Correlation of specific PID with given feature set (voice,ES)
        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.scatter(mv['mnh'], mv[f])
        ax1.set_title("Sex %d, Age %d - %d, mnh-%s, R = %1.2f" %(sex, age_min, age_max, f, R))
        ax1.set_xlabel("%s" %f)
        ax1.set_ylabel("Mac New Heart")
        mv = mv.groupby("date").mean()
        ax2.plot(t, mv['mnh'], label='mnh')
        ax2.plot(t, mv[f], label=f)
        ax2.set_xlabel("Day")
        plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))
        #plt.tight_layout()
        plt.show()
        # , label = mv.columns['mnh']
    def correl_mnh_vf():
        demo_dir = os.path.expanduser(os.path.join("~", "polybox", "ETH_Master_Arbeit", "CAMP"))
        df_demo = pd.read_csv("%s/PID_info.csv" %demo_dir)#, index_col='PID') ## Import dataframe
        t = ['1', '2', 'B']#, 'C', 'D']
        for sex in range(1,3):# = 1 # 1=male, 2=female
            for age_min in range(40,70,40):
                # if age_min >= 60:
                #     age_max = age_min + 19 + 11
                # else:
                #     age_max = age_min + 19
                age_max = age_min + 39
                # age_min = df_demo['age'].min()
                # age_max = df_demo['age'].max()
                df_gender = df_demo[(df_demo['sex'] == sex)]
                df_age = df_gender[(df_demo['age'] >= age_min) & (df_demo['age'] <= age_max)]
                mv = df_mnh_vf(df_MNH, df_vf,df_age)  ## uncomment correl_mnh_vf in the end of the function to see correlation
                mv = mv[mv.date != '1']
                t = ['2', 'B']

                for f in mv.columns[5:]:
                    corr_f = pg.corr(mv['mnh'], mv[f]) # or df.corr(method='pearson')
                    R = corr_f.iloc[0,1]
                    if abs(R) >= 0.5:
                        correl_mnh_plot_all(mv,t, R, f, sex, age_min, age_max)


    start = time.time()
    rootdir_audio = os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "CAMP_study_data", "Emotional_recordings_audio")) #, "549088_EmotionalRecordings"))
    patient_folders = [] # # Get list of all patient folders in rootdir that have a minimum of x recordings
    for subdir, dirs, files in os.walk(rootdir_audio):
        for folder in dirs:
            number = len(os.listdir(os.path.join(rootdir_audio, folder)))
            #print(folder, number)
            if number > 1:
                patient_folders.append(folder)#[:7])
    patient_folders = list(map(int, patient_folders))
    patient_folders.sort()
    df_MNH, mnh_X = MNH_load()
    df = df_MNH
    df_vf, pat = vf_load()

    #MNH_plots(patient_folders, df, mnh_X)
    #MNH_stats_basic(df)
    #MNH_stats_ttest_dep(df)
    #MNH_stats_ttest_indep(df)
    #te = pg.ttest(df['mnh_2_sum'], df['mnh_B_sum'], paired=True)
    #normality(df)
    correl_mnh_vf()

    end = time.time()
    print((end-start)/60)
