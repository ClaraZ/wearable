
# coding: utf-8

# In[3]:


import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


pd.set_option('display.max_columns',None)

matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize']=[12.0,6.0]


# In[6]:


# Importing the dataset
filename="30-day sleep.csv"
data = pd.read_csv(filename,converters={'Date':pd.to_datetime})
data.set_index(pd.to_datetime(data.Date),inplace=True)
print('loaded db successfully!')


# In[7]:


data.head()


# In[8]:


# Creating new columns and cleaning data
## first remove the activity where no steps were recorded. For sleep data, remove rows where there was no 'deep sleep' entries
dayCodes=['','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
minDayCodes=['',',Mon','Tue','Wed','Thu','Fri','Sat','Sun']
days={1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}


# In[9]:


def is_nan(x):
    return (x is np.nan or x!=x)

def defineSleepBucket(row):
    sleepEntry=row['SleepStart']
    if not is_nan(sleepEntry):
        sleepTimeO=datetime.datetime.strptime(sleepEntry,'%Y-%m-%d %H:%M').time()
        if sleepTimeO.minute>30:
            return(sleepTimeO.hour+1)
        elif sleepTimeO.minute>0:
            return sleepTimeO.hour+0.5
    else:
        return np.nan
    
def defineAwakeBucket(row):
    awakeEntry=row['SleepEnd']
    if not is_nan(awakeEntry):
        awakeEntryO=datetime.datetime.strptime(awakeEntry,'%Y-%m-%d %H:%M').time()
        if awakeEntryO.minute>30:
            return (awakeEntryO.hour+1)
        elif awakeEntryO.minute>0:
            return awakeEntryO.hour+0.5
    else:
        return np.nan


# In[25]:


#remove all entries where there was no steps recorded i.e. no activity
data=data[data['DailySteps']!=0]
#remove all entries with no deep sleep recorded
sleepData=data[data['DeepSleepMin']!=0]
import time
from datetime import date
data['Day of Week']=data['Date'].apply(lambda x: x.isoweekday())
data['Day Label']=data['Day of Week'].apply(lambda x:days[x])


# In[12]:


#build additional sleep columns
data['Sleep Bucket']=data.apply(defineSleepBucket,axis=1)
data['Awake Bucket']=data.apply(defineAwakeBucket,axis=1)


# In[13]:


#redefine InBedMin as DeepSleepMin+LightSleepMin+AwakeMin
data['InBedMin']=data['DeepSleepMin']+data['LightSleepMin']+data['AwakeMin']
#feature constructions
data['% Deep sleep']=100*data['DeepSleepMin']/data['InBedMin']
data['% Light sleep']=100*data['LightSleepMin']/data['InBedMin']
data['% Awake']=100-(data['% Deep sleep']+data['% Light sleep'])
data['Is Weekday']=data['Day of Week'].isin ([1,2,3,4,5])
data['Is Weekend']=data['Day of Week'].isin ([6,7])
data['Minutes Sedentary']=1440-(data['InBedMin']+data['RunTimeMin']+data['WalkTimeMin'])
data.head()


# In[14]:


## weekdays vs weekends
dayGroupedData=data.groupby(['Day of Week']).mean()
dayTypeGroupedData=data.groupby(['Is Weekday']).mean()
dayTypeGroupedData
dayGroupedData


# In[15]:


# Utilities
def getDayLabel(dayNum):
    return dayCodes[dayNum]

def plot_heatmap(corrmat,correlationOf,title,darkTheme=False):
    if darkTheme:
        sns.set(style='darkgrid',palette='deep') #using seaborn for making heatmap
        cmap='YlGnBu'
    else:
        sns.set(style='white')
        cmap=sns.diverging_palette(220,10,as_cmap=True)
    # Generate a mask for the upper triangle
    mask=np.zeros_like(corrmat,dtype=np.bool)
    mask[np.triu_indices_from(mask)]=True
    
    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(10,10))
    hm=sns.heatmap(corrmat,mask=mask,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},cmap=cmap)
    hm.set_title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[17]:


# Activity analysis

## 1. Activity summary - steps, calories
data[['DailyDistanceMeter','DailyBurnCalories','DailySteps','Minutes Sedentary','WalkDistance','WalkTimeMin','WalkBurnCalories','RunDistanceMeter','RunTimeMin','RunBurnCalories','WalkRunSeconds']].describe().transpose()


# In[18]:


fig=plt.figure(figsize=(20,6))

ax=plt.subplot(131)
plt.bar(dayGroupedData.index,dayGroupedData['DailySteps'])
plt.title('Day of Week vs. DailySteps',fontsize=15)
plt.xlabel('Day of Week',fontsize=14)
plt.ylabel("DailySteps",fontsize=14)
ax.axhline(350,color='orangered',linestyle='--')
ax.axhline(2500,color='orange',linestyle='--')
ax.set_xticklabels(minDayCodes)

###########
ax2=fig.add_subplot(132)
plt.bar(dayGroupedData.index,dayGroupedData['DailyBurnCalories'],color='blueviolet')
plt.title('Day of Week vs. DailyBurnCalories',fontsize=15)
plt.xlabel('Day of Week',fontsize=14)
plt.ylabel('DailyBurnCalories',fontsize=14)
ax2.set_xticklabels(minDayCodes)

###########
ax3=fig.add_subplot(133)
ax3.set_xticklabels(minDayCodes)
plt.bar(dayGroupedData.index,dayGroupedData['DailyDistanceMeter'],color='orange')
plt.title('Day of Week vs. DailyDistanceMeter',fontsize=15)
plt.xlabel('Day of Week',fontsize=14)
plt.ylabel("DailyDistanceMeter",fontsize=14)
plt.show()


# In[19]:


# 2. Sedentary minutes

plt.bar(dayGroupedData.index,dayGroupedData['Minutes Sedentary'],color='orange',tick_label=minDayCodes[1:])
plt.title('Sedentary minutes per day')


# In[21]:


## Calorie burn correlation
correlationOf='DailyBurnCalories'
corrdf_calories=data[['DailyBurnCalories','DailyDistanceMeter','DailySteps','Minutes Sedentary','WalkDistance','WalkTimeMin','WalkBurnCalories','RunDistanceMeter','RunTimeMin','RunBurnCalories','WalkRunSeconds','Is Weekday']]
plot_heatmap(corrdf_calories.corr(),correlationOf,'')


# In[22]:


# Basic correlogram
sns.pairplot(corrdf_calories.dropna(),kind='scatter',markers='+',plot_kws=dict(s=50,edgecolor='b',linewidth=1))
plt.show()


# In[23]:


# the calories burned is strongly related to steps,distance,and WalkRunSeconds. Minutes sedentary has a negative correlation with Weekdays which implies spending more time slacking off on weekends


# In[ ]:


# Sleep Analysis
## 1. How regular is the sleeping habit?
## getting the required hours of sleep? -> average sleep hours and the deviation
## following a good sleep schedule? -> average sleep and wake up timings


# In[34]:


import matplotlib.dates as mdates

sleepDesc=pd.DataFrame(sleepData['InBedMin']/60).describe().transpose()
avgSleepHours=round(sleepDesc.at['InBedMin','mean'],2)
summary='Averaging a sleep of {} hours with a deviation of {} hours'.format(avgSleepHours,round(sleepDesc.at['InBedMin','std'],2))
hourInBed=sleepData['InBedMin']/60

fig=plt.figure(figsize=(20,6))

ax=plt.subplot(121)
plt.hist(hourInBed,bins=8,range=(3,10),color='navy')
plt.xlim(3,10)
plt.xticks(range(3,10))
plt.xlabel('hour in bed')
plt.ylabel('count');
plt.title(summary,fontsize=15)

############
ax2 = fig.add_subplot(122)
plt.plot(sleepData['Date'],hourInBed, linestyle='-', markersize=10, color='darkturquoise', label='% Light', linewidth=3.0, alpha=0.9)
plt.ylabel('InBedMin', fontsize=14)
ax2.axhline(avgSleepHours, color="orangered", linestyle='--')
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
ax2.grid(True)
plt.xticks(rotation=75)
plt.plot()

sleepDesc


# In[35]:


sleepBDF=sleepData[['Sleep Bucket','Awake Bucket','InBedMin']]
sleepBDF['InBedMin']=sleepBDF['InBedMin']/60

#sleepBDF.groupby(['Sleep Bucket']).mean()
#sleepBDF.describe().transpose()

## plot the sleep and awake counts
fig=plt.figure(figsize=(20,6))

ax=plt.subplot(121)
pd.value_counts(sleepData['Sleep Bucket']).plot.bar(cmap='BuPu_r',alpha=0.8)
plt.xlabel('time of sleep',fontsize=14)
plt.ylabel("counts",fontsize=14)
plt.xticks(rotation=0)

#############
ax2=fig.add_subplot(122)
pd.value_counts(sleepData['Awake Bucket']).plot.bar(cmap='plasma',alpha=0.5)
plt.xlabel('time to wake up',fontsize=14)
plt.ylabel('counts',fontsize=14)
plt.xticks(rotation=0)
plt.show()
#https://www.sleepfoundation.org/sleep-tools-tips/healthy-sleep-tips


# In[36]:


sleepBDF_weekday = sleepData[['Sleep Bucket', 'Awake Bucket', 'InBedMin', 'Is Weekday']]
sleepBDF_weekday['InBedMin'] = sleepBDF_weekday['InBedMin']/60
sleepBDF_weekday = sleepBDF_weekday[sleepBDF_weekday['Is Weekday']]

#sleepBDF.groupby(['Sleep Bucket']).mean()
#sleepBDF.describe().transpose()

## plot the sleep and awake counts
fig = plt.figure(figsize = (20,6))

ax = plt.subplot(121)  
pd.value_counts(sleepBDF_weekday['Sleep Bucket']).plot.bar(cmap="BuPu_r", alpha=0.8)
plt.xlabel('Time of sleep', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.xticks(rotation=0)

#############

ax2 = fig.add_subplot(122)
pd.value_counts(sleepBDF_weekday['Awake Bucket']).plot.bar(cmap="plasma", alpha=0.5)
plt.xlabel('Time to wake up', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.xticks(rotation=0)
plt.show()

#https://www.sleepfoundation.org/sleep-tools-tips/healthy-sleep-tips


# In[37]:


# 2. Types of sleep
avgSleep=sleepData[['LightSleepMin','DeepSleepMin','AwakeMin']].mean()

fig=plt.figure(figsize=(6,6))
labels=['light sleep','deep sleep','awake']
plt.pie(avgSleep,colors=['darkturquoise','salmon','lightskyblue','yellowgreen'],autopct='%1.1f%%',labels=labels,textprops=dict(color='w'))

##carve the donut
# my_circle=plt.Circle((0,0),0.7,color='white')
# p=plt.gcf()
# p.gca().add_artist(my_circle)

plt.title('Average of types of sleep',fontsize=14)
plt.legend()
plt.show()


# In[38]:


plt.plot(sleepData['Date'],sleepData['LightSleepMin'], linestyle='-', markersize=10, color='darkturquoise', label='Minutes Light', linewidth=3.0, alpha=0.9)
plt.plot(sleepData['Date'],sleepData['DeepSleepMin'], linestyle='--', markersize=10, color='red', label='Minutes Deep', linewidth=3.0, alpha=0.9)
plt.legend()


# In[39]:


# 3. Correlation between amount of sleep and the sleep stages
## does sleeping more will help attain more deep sleep?

corrdf_sleep_types=sleepData[['InBedMin','DeepSleepMin','LightSleepMin','AwakeMin']].corr().abs()
plot_heatmap(corrdf_sleep_types,correlationOf,'Correlation of time in bed with other sleep stages')
corrdf_sleep_types


# In[40]:


## deep sleep minutes are very correlated with time in bed, so sleeping more might guarantee a good deep sleep but no medical evidence yet


# In[41]:


# 4. Types of sleep based on different days
fig = plt.figure(figsize = (14,5))
plt.bar((dayGroupedData.index), dayGroupedData['LightSleepMin'],width = 0.4, color='lightskyblue', label="Minutes Light sleep", tick_label=minDayCodes[1:])
plt.bar((dayGroupedData.index + 0.1), dayGroupedData['AwakeMin'], width = 0.4, color='dodgerblue', label="AwakeMin")
plt.bar((dayGroupedData.index + 0.2), dayGroupedData['DeepSleepMin'], width = 0.4, color='slateblue', label="Minutes Deep sleep")
plt.xlabel('Day of Week')
plt.ylabel('Sleep stages')
plt.legend()


# In[42]:


# 5. Effect of Sleep on weekdays vs weekends
ax = sleepData.boxplot(column = 'InBedMin', by = 'Is Weekend', vert = False, widths = 0.4)
plt.xlabel('Minutes in Bed')
plt.suptitle('')
plt.title('');

## The below plot shows that the person tends to sleep less on Weekends. The upper whisker is quite high for weekends indicatign varying sleep times.
## let's check out how does the plots vary for indivisual days of the week.


# In[43]:


ax = sleepData.boxplot(column = 'InBedMin', by = 'Day of Week')
ax.set_xticklabels(minDayCodes[1:])
plt.ylabel('Minutes in Bed')
plt.suptitle('')
plt.title('');

# need more data, the deviation is too big across the 5 weekdays


# In[51]:


# correlation of sleep with other physical activity
sleepData['8 > Sleep > 7'] = sleepData['InBedMin'] > 7*60
sleepData['Sleep > 7'] = sleepData['InBedMin'] > 7*60
sleepData['Sleep > 8'] = sleepData['InBedMin'] > 8*60
sleepData['Active mins > 30'] = sleepData['WalkRunSeconds'] > 1800
sleepData['Active mins > 60'] = sleepData['WalkRunSeconds'] > 3600

sleepData['wee'] = np.logical_and(sleepData['Sleep Bucket'] <= 23, sleepData['Awake Bucket'] <= 6.5)
# slept before 11 and woke up by 6:30
#sleepData


# In[52]:


correlationOf="DeepSleepMin"
k = 17 #number of variables for heatmap

corrmat = sleepData[['DeepSleepMin', 'DailyBurnCalories', 'DailyDistanceMeter', 'DailySteps', 'Minutes Sedentary','WalkDistance','WalkTimeMin','WalkBurnCalories','RunDistanceMeter','RunTimeMin','RunBurnCalories','WalkRunSeconds','8 > Sleep > 7', 'wee', 'Sleep > 8', 'Active mins > 60', 'Active mins > 30']].corr()

#corrmat = sleepData.drop(['% Restorative sleep', 'Minutes Light sleep', 'Minutes REM sleep', '% Deep sleep', '% Light sleep', '% REM sleep', 'REM sleep count', 'Deep sleep count', 'Light sleep count'], axis=1).corr().abs()
#cols = corrmat.nlargest(k, correlationOf)[correlationOf].index
#corrdf_sleep = sleepData[cols]

plot_heatmap(corrmat, correlationOf, 'Coorelation of {} with others'.format(correlationOf), darkTheme=False)


# In[53]:


## deep sleep is strongly correclated with 7~8 hours' sleep and sedentary minutes


# In[54]:


# Machine Learning
# Since we have some insights now on PA and Sleep. Let's use some basic ML techniques to see if there is a pattern to predict what are the ingredients for a decent sleep!
# But we have few data and very few features, I don't expect to see some magic here. Will be repeating this once i have some more sleep data.
# For now I could lay the basic template.

