#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading the required libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


#loading the ipl matches dataset
ipl=pd.read_csv('matches.csv')
deliveries=pd.read_csv('deliveries.csv')


# In[3]:


#having a glance at the first Ten records of the dataset
ipl.head(10)


# In[4]:


deliveries.head()


# In[5]:


#Lookin at the number of rows and columns in the dataset
ipl.shape


# In[6]:


#Analysis of Teams


# In[7]:


#Looking at the number of matches played each season
matches = ipl['season'].value_counts()


# In[8]:


plt.figure(figsize=(12,6))
sns.barplot(x=ipl['season'].value_counts().keys(), y=ipl['season'].value_counts().values,  palette="RdBu")
plt.title("Number of matches played each season")
plt.show()


# In[9]:


#Getting the frequency of result column
print("The frequency of result of Matches :")
print(ipl['result'].value_counts())


# In[10]:


#Finding out the number of toss wins w.r.t each team
print("Number of toss wins w.r.t each team :")
print(ipl['toss_winner'].value_counts())


# In[11]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=ipl['toss_winner'].value_counts().keys(), y=ipl['toss_winner'].value_counts().values,  palette="PuRd")
plt.title("Number of toss wins w.r.t each team")
plt.show()


# In[12]:


#Extracting the records where a team won batting first
batting_first=ipl[ipl['win_by_runs']!=0]


# In[13]:


#Looking at the head
batting_first.head(10)


# In[14]:


plt.figure(figsize=(15,10))
plt.hist(batting_first['win_by_runs'],color='red')
plt.title("Histogram for frequency of match wins w.r.t runs")
plt.xlabel("Runs")
plt.ylabel("Number of Matches")
plt.show()


# In[15]:


#Finding out the number of wins w.r.t each team after batting first
print("The number of wins w.r.t each team after batting first :")
print(batting_first['winner'].value_counts())


# In[16]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=batting_first['winner'].value_counts().keys(), y=batting_first['winner'].value_counts().values,  palette="YlGnBu")
plt.title("Teams win match after batting first")
plt.show()


# In[17]:


plt.figure(figsize=(12,12))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.title("Pie chart for distribution of most wins after batting first")
plt.show()


# In[18]:


#extracting those records where a team has won after batting second
batting_second=ipl[ipl['win_by_wickets']!=0]


# In[19]:


batting_second.head(10)


# In[20]:


plt.figure(figsize=(15,10))
plt.hist(batting_second['win_by_wickets'],bins=20,color='red')
plt.title("Histogram for frequency of wins w.r.t number of wickets")
plt.xlabel("Wickets")
plt.ylabel("Number of Matches wins")
plt.show()


# In[21]:


print("Teams win match after batting second :")
print(batting_second['winner'].value_counts())


# In[22]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=batting_second['winner'].value_counts().keys(), y=batting_second['winner'].value_counts().values,  palette="PuOr")
plt.title("Teams win match after batting second")
plt.show()


# In[23]:


plt.figure(figsize=(12,12))
plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.title("Pie chart for distribution of most wins after batting second")
plt.show()


# In[24]:


matches_played=pd.concat([ipl['team1'],ipl['team2']])
matches_played=matches_played.value_counts().reset_index()
matches_played.columns=['Team','Total Matches']
matches_played['wins']=ipl['winner'].value_counts().reset_index()['winner']

matches_played.set_index('Team',inplace=True)


# In[25]:


print("Winning record of each team :")
print(matches_played.reset_index().head(10))


# In[26]:


win_percentage = round(matches_played['wins']/matches_played['Total Matches'],3)*100
print("Winning percentage of each team :")
print(win_percentage.head(5))


# In[27]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=win_percentage[0:10].keys(), y=win_percentage[0:10].values,  palette="PiYG")
plt.title("Win percentage of each team ")
plt.show()


# In[28]:


#Looking at the number of matches played in each city
print("The number of matches played in each city :")
print(ipl['city'].value_counts())


# In[29]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=ipl['city'].value_counts().keys(), y=ipl['city'].value_counts().values,  palette="YlGnBu")
plt.title("Number of matches played in each city")
plt.show()


# In[30]:


print("Number of matches played in each stadium :")
print(ipl['venue'].value_counts())


# In[31]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=ipl['venue'].value_counts().keys(), x=ipl['venue'].value_counts().values,  palette="PRGn")
plt.title("Number of matches played in each stadium")
plt.show()


# In[32]:


high_scores = deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 
high_scores = high_scores[high_scores['total_runs']>=200]
print("Highest total score of team in an inning :")
print(high_scores.nlargest(10,'total_runs'))


# In[33]:


team_extra = deliveries.groupby('bowling_team').apply(lambda x : sum(x['extra_runs'])).reset_index(name='Extra Runs')
team_extra_sorted=team_extra.sort_values(by='Extra Runs',ascending=False)
top_team_extra=team_extra_sorted[0:15]
print("Teams given total Extra runs :")
print(top_team_extra)


# In[34]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_team_extra['Extra Runs'], x=top_team_extra['bowling_team'],  palette="Oranges")
plt.title("Teams given total Extra runs")
plt.show()


# In[35]:


team_runs = deliveries.groupby('batting_team').apply(lambda x : sum(x['total_runs'])).reset_index(name='Total Runs')
team_runs_sorted=team_runs.sort_values(by='Total Runs',ascending=False)
top_team_runs=team_runs_sorted[0:15]
print("Teams total runs scored in 12 seasons :")
print(top_team_runs)


# In[36]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_team_runs['Total Runs'], x=top_team_runs['batting_team'],  palette="Reds")
plt.title("Teams total runs scored in 12 seasons")
plt.show()


# In[37]:


Team_scored_six = deliveries[deliveries['batsman_runs'] == 6]
Team_scored_six_total = Team_scored_six.groupby('batting_team').apply(lambda x : x['batsman_runs'].dropna()).reset_index(name='Six')
Team_scored_six_count = Team_scored_six_total.groupby('batting_team').count().reset_index()
Team_scored_six_top = Team_scored_six_count.sort_values(by='Six',ascending=False)
top_Team_scored_six = Team_scored_six_top.loc[:,['batting_team','Six']][0:15]
print("Teams with number of sixes scored :")
print(top_Team_scored_six)


# In[38]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Team_scored_six['Six'], x=top_Team_scored_six['batting_team'],  palette="Blues")
plt.title("Teams with number of sixes scored  ")
plt.show()


# In[39]:


#Teams with number of fours scored
Team_scored_fours = deliveries[deliveries['batsman_runs'] == 4]
Team_scored_fours_total = Team_scored_fours.groupby('batting_team').apply(lambda x : x['batsman_runs'].dropna()).reset_index(name='Fours')
Team_scored_fours_count = Team_scored_fours_total.groupby('batting_team').count().reset_index()
Team_scored_fours_top = Team_scored_fours_count.sort_values(by='Fours',ascending=False)
top_Team_scored_fours = Team_scored_fours_top.loc[:,['batting_team','Fours']][0:15]
print("Teams with number of fours scored :")
print(top_Team_scored_fours)


# In[40]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Team_scored_fours['Fours'], x=top_Team_scored_fours['batting_team'],  palette="RdYlGn")
plt.title("Teams with number of fours scored  ")
plt.show()


# In[41]:


#Teams with number of wickets taken
Team_total = deliveries.groupby('bowling_team').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
Team_count = Team_total.groupby('bowling_team').count().reset_index()
Team_top = Team_count.sort_values(by='Wickets',ascending=False)
top_Team = Team_top.loc[:,['bowling_team','Wickets']][0:15]
print("Teams with number of wickets taken :")
print(top_Team)


# In[42]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Team['Wickets'], x=top_Team['bowling_team'],  palette="RdYlBu")
plt.title("Teams with number of wickets taken  ")
plt.show()


# In[43]:


#players data analysis


# In[44]:


#Getting the frequency of most man of the match awards
ipl['player_of_match'].value_counts()


# In[45]:


#Getting the top 10 players with most man of the match awards
print("Top 20 players with most man of the match awards :")
print(ipl['player_of_match'].value_counts()[0:20])


# In[46]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=ipl['player_of_match'].value_counts()[0:20].keys(), y=ipl['player_of_match'].value_counts()[0:20].values,  palette="magma")
plt.title("Top 20 players with most man of the match awards")
plt.show()


# In[47]:


batting_ings = deliveries.groupby(['match_id','batsman']).apply(lambda x : sum(x['batsman_runs'])).reset_index(name='Innings Runs')
sorted_batting_ings = batting_ings.sort_values(by='Innings Runs',ascending=False)
top_batsmen_scores = sorted_batting_ings[:10] 
ball_faced = deliveries.groupby(['match_id','batsman']).apply(lambda x : x['batsman_runs'].count()).reset_index(name='Balls Faced')
batsmen_performance = pd.merge(top_batsmen_scores, ball_faced, how='inner', left_on=['match_id','batsman'], right_on=['match_id','batsman'])
batsmen_performance['Strike Rate for Match'] = batsmen_performance['Innings Runs']*100 / batsmen_performance['Balls Faced']
batsmen_innings = pd.merge(batsmen_performance, deliveries, how='inner',left_on=['match_id','batsman'],right_on=['match_id','batsman'])
batsmen_innings_table = batsmen_innings.iloc[:,1:8]
batsmen_innings_table2 = batsmen_innings_table.drop_duplicates()
print("Most runs scored by a batsman in a one inning :")
print(batsmen_innings_table2)


# In[48]:


x=batsmen_innings_table2['batsman']
y1=batsmen_innings_table2['Innings Runs']
plt.figure(figsize=(12,6))
plt.scatter(x,y1)
plt.xlabel('Batsmen',size=15)
plt.ylabel('Innings Score',size=15)
plt.title('IPL Best batting performances in an Inning')
plt.xticks(rotation=90)
plt.legend(['Runs']);


# In[49]:


bowling_wickets = deliveries[deliveries['dismissal_kind']!='run out']
bowling_total = bowling_wickets.groupby('bowler').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
bowling_wicket_count = bowling_total.groupby('bowler').count().reset_index()
bowling_top = bowling_wicket_count.sort_values(by='Wickets',ascending=False)
top_bowlers = bowling_top.loc[:,['bowler','Wickets']][0:20]
print("Top 20 Bowlers with most Wickets :")
print(top_bowlers)


# In[50]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_bowlers['Wickets'], x=top_bowlers['bowler'],  palette="PiYG")
plt.title("Top 20 Bowlers with most Wickets")
plt.show()


# In[51]:


plt.figure(figsize=(12,6))
plt.scatter(top_bowlers['bowler'],top_bowlers['Wickets'],color='r');
plt.plot(top_bowlers['bowler'],top_bowlers['Wickets'],color='g');
plt.xticks(rotation=90)
plt.xlabel('Top 10 Bowlers')
plt.ylabel('Wickets Taken')
plt.title('Top 10 Bowlers in last 12 seasons');


# In[52]:


Run_out_wickets = deliveries[deliveries['dismissal_kind'] =='run out']
Run_out_total = Run_out_wickets.groupby('fielder').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
Run_out_count = Run_out_total.groupby('fielder').count().reset_index()
Run_out_top = Run_out_count.sort_values(by='Wickets',ascending=False)
top_fielders = Run_out_top.loc[:,['fielder','Wickets']][0:20]
print("Top 20 fielders with most Run outs :")
print(top_fielders)


# In[53]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_fielders['Wickets'], x=top_fielders['fielder'],  palette="YlGnBu")
plt.title("Top 20 fielders with most Run outs")
plt.show()


# In[54]:


bowled_out_wickets = deliveries[deliveries['dismissal_kind'] =='bowled']
bowled_out_total = bowled_out_wickets.groupby('bowler').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
bowled_out_count = bowled_out_total.groupby('bowler').count().reset_index()
bowled_out_top = bowled_out_count.sort_values(by='Wickets',ascending=False)
top_bowler = bowled_out_top.loc[:,['bowler','Wickets']][0:20]
print("Top 20 bowlers with most bowled outs the batsman :")
print(top_bowler)


# In[55]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_bowler['Wickets'], x=top_bowler['bowler'],  palette="YlGnBu")
plt.title("Top 20 bowlerss with most bowled outs the batsman")
plt.show()


# In[56]:


Bowler_delivered_no_balls = deliveries[deliveries['noball_runs'] == 1]
Bowler_delivered_no_balls_total = Bowler_delivered_no_balls.groupby('bowler').apply(lambda x : x['noball_runs'].dropna()).reset_index(name='No balls')
Bowler_delivered_no_balls_count = Bowler_delivered_no_balls_total.groupby('bowler').count().reset_index()
Bowler_delivered_no_balls_top = Bowler_delivered_no_balls_count.sort_values(by='No balls',ascending=False)
top_Bowler_delivered_no_balls = Bowler_delivered_no_balls_top.loc[:,['bowler','No balls']][0:20]
print("Top 20 bowlers with scored most no balls :")
print(top_Bowler_delivered_no_balls)


# In[57]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Bowler_delivered_no_balls['No balls'], x=top_Bowler_delivered_no_balls['bowler'],  palette="YlOrBr")
plt.title("Top 20 bowlers with scored most no balls ")
plt.show()


# In[58]:


#players take most catches
catches_taken = deliveries[deliveries['dismissal_kind'] =='caught']
catches_taken_total = catches_taken.groupby('fielder').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
catches_taken_count = catches_taken_total.groupby('fielder').count().reset_index()
catches_taken_top = catches_taken_count.sort_values(by='Wickets',ascending=False)
top_fielders = catches_taken_top.loc[:,['fielder','Wickets']][0:20]
print("Players with number of catches taken :")
print(top_fielders)


# In[59]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_fielders['Wickets'], x=top_fielders['fielder'],  palette="RdGy")
plt.title("Players with number of catches taken  ")
plt.show()


# In[60]:


batsmen = deliveries.groupby('batsman').apply(lambda x : sum(x['batsman_runs'])).reset_index(name='Runs')
batsmen_sorted=batsmen.sort_values(by='Runs',ascending=False)
top_batsmen=batsmen_sorted[0:20]
print("Top 20 Batsmen scored most runs :")
print(top_batsmen)


# In[61]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsmen['Runs'], x=top_batsmen['batsman'],  palette="YlOrRd")
plt.title("Top 20 Batsmen scored most runs")
plt.show()


# In[62]:


Batsman_scored_six = deliveries[deliveries['batsman_runs'] == 6]
Batsman_scored_six_total = Batsman_scored_six.groupby('batsman').apply(lambda x : x['batsman_runs'].dropna()).reset_index(name='Six')
Batsman_scored_six_count = Batsman_scored_six_total.groupby('batsman').count().reset_index()
Batsman_scored_six_top = Batsman_scored_six_count.sort_values(by='Six',ascending=False)
top_Batsman_scored_six = Batsman_scored_six_top.loc[:,['batsman','Six']][0:20]
print("Top 20 batsman scored most six :")
print(top_Batsman_scored_six)


# In[63]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Batsman_scored_six['Six'], x=top_Batsman_scored_six['batsman'],  palette="PRGn")
plt.title("Top 20 batsman scored most six ")
plt.show()


# In[64]:


Batsman_scored_fours = deliveries[deliveries['batsman_runs'] == 4]
Batsman_scored_fours_total = Batsman_scored_fours.groupby('batsman').apply(lambda x : x['batsman_runs'].dropna()).reset_index(name='Fours')
Batsman_scored_fours_count = Batsman_scored_fours_total.groupby('batsman').count().reset_index()
Batsman_scored_fours_top = Batsman_scored_fours_count.sort_values(by='Fours',ascending=False)
top_Batsman_scored_fours = Batsman_scored_fours_top.loc[:,['batsman','Fours']][0:20]
print("Top 20 batsman scored most fours :")
print(top_Batsman_scored_fours)


# In[65]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Batsman_scored_fours['Fours'], x=top_Batsman_scored_fours['batsman'],  palette="PiYG")
plt.title("Top 20 batsman scored most fours ")
plt.show()


# In[66]:


batsmen1 = deliveries.groupby('batsman').apply(lambda x : sum(x['ball'])).reset_index(name='Balls faced')
batsmen1_sorted=batsmen1.sort_values(by='Balls faced',ascending=False)
top_batsmen1=batsmen1_sorted[0:20]
print("Top 20 Batsmen faced most balls :")
print(top_batsmen1)


# In[67]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsmen1['Balls faced'], x=top_batsmen1['batsman'],  palette="BrBG")
plt.title("Top 20 Batsmen faced most balls")
plt.show()


# In[68]:


Batsman_played_dot_balls = deliveries[deliveries['batsman_runs'] == 0]
Batsman_played_dot_balls_total = Batsman_played_dot_balls.groupby('batsman').apply(lambda x : x['batsman_runs'].dropna()).reset_index(name='Dot balls')
Batsman_played_dot_balls_count = Batsman_played_dot_balls_total.groupby('batsman').count().reset_index()
Batsman_played_dot_balls_top = Batsman_played_dot_balls_count.sort_values(by='Dot balls',ascending=False)
top_Batsman_played_dot_balls = Batsman_played_dot_balls_top.loc[:,['batsman','Dot balls']][0:20]
print("Top 20 batsman played most dot balls :")
print(top_Batsman_played_dot_balls)


# In[69]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_Batsman_played_dot_balls['Dot balls'], x=top_Batsman_played_dot_balls['batsman'],  palette="YlGn")
plt.title("Top 20 batsman played most dot balls ")
plt.show()


# In[70]:


batting_ings = deliveries.groupby(['match_id','batsman']).apply(lambda x : sum(x['batsman_runs'])).reset_index(name='Innings Runs')
sorted_batting_ings = batting_ings.sort_values(by='Innings Runs',ascending=False)
top_batsmen_scores = sorted_batting_ings[0:]
top_batsmen_scores = top_batsmen_scores[top_batsmen_scores['Innings Runs']>=100]
print("Players scored number of century :")
print(top_batsmen_scores['batsman'].value_counts())


# In[71]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsmen_scores['batsman'].value_counts()[0:20].values, x=top_batsmen_scores['batsman'].value_counts()[0:20].keys(),  palette="RdBu")
plt.title("Players scored number of century ")
plt.show()


# In[72]:


batting_ings = deliveries.groupby(['match_id','batsman']).apply(lambda x : sum(x['batsman_runs'])).reset_index(name='Innings Runs')
sorted_batting_ings = batting_ings.sort_values(by='Innings Runs',ascending=False)
top_batsmen_scores = sorted_batting_ings[0:]
top_batsmen_scores =top_batsmen_scores[top_batsmen_scores['Innings Runs']>=50]
print("Players score number of times scored fifty+ runs :")
print(top_batsmen_scores['batsman'].value_counts())


# In[73]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsmen_scores['batsman'].value_counts()[0:20].values, x=top_batsmen_scores['batsman'].value_counts()[0:20].keys(),  palette="PuOr")
plt.title("Players score number of times scored fifty+ runs ")
plt.show()


# In[74]:


Run_out_wickets1 = deliveries[deliveries['dismissal_kind'] =='run out']
Run_out_total1 = Run_out_wickets1.groupby('batsman').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
Run_out_count1 = Run_out_total1.groupby('batsman').count().reset_index()
Run_out_top1 = Run_out_count1.sort_values(by='Wickets',ascending=False)
top_batsman1 = Run_out_top1.loc[:,['batsman','Wickets']][0:20]
print("Top 20 batsman with most time Run outs :")
print(top_batsman1)


# In[75]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsman1['Wickets'], x=top_batsman1['batsman'],  palette="YlGnBu")
plt.title("Top 20 batsmans with most time Run outs")
plt.show()


# In[76]:


bowled_out_wickets1 = deliveries[deliveries['dismissal_kind'] =='bowled']
bowled_out_total1 = bowled_out_wickets1.groupby('batsman').apply(lambda x : x['dismissal_kind'].dropna()).reset_index(name='Wickets')
bowled_out_count1 = bowled_out_total1.groupby('batsman').count().reset_index()
bowled_out_top1 = bowled_out_count1.sort_values(by='Wickets',ascending=False)
top_batsman2 = bowled_out_top1.loc[:,['batsman','Wickets']][0:20]
print("Top 20 batsman with most time bowled outs :")
print(top_batsman2)


# In[77]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsman2['Wickets'], x=top_batsman2['batsman'],  palette="RdBu")
plt.title("Top 20 batsmans with most time bowled outs")
plt.show()


# In[88]:


batting_ings = deliveries.groupby(['match_id','batsman']).apply(lambda x : sum(x['batsman_runs'])).reset_index(name='Innings Runs')
sorted_batting_ings = batting_ings.sort_values(by='Innings Runs',ascending=False)
top_batsmen_scores = sorted_batting_ings[0:]
top_batsmen_scores = top_batsmen_scores[top_batsmen_scores['Innings Runs'] == 0]
print("Players with most times duck outs(Out on score zero) :")
print(top_batsmen_scores['batsman'].value_counts())


# In[89]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(y=top_batsmen_scores['batsman'].value_counts()[0:20].values, x=top_batsmen_scores['batsman'].value_counts()[0:20].keys(),  palette="PuOr")
plt.title("Players with most times duck outs(Out on score zero) ")
plt.show()


# In[78]:


print("Number of matches person stand as a umpire :")
print(ipl['umpire1'or'umpire2'or'umpire3'].value_counts())


# In[79]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(x=ipl['umpire1'or'umpire2'or'umpire3'].value_counts().keys(), y=ipl['umpire1'or'umpire2'or'umpire3'].value_counts().values,  palette="RdBu")
plt.title("Number of matches person stand as a umpire")
plt.show()

