import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("特征提取与工程处理")
print("=" * 80)

# ============== 读取清洗后的数据 ==============
athletes_df = pd.read_csv('summerOly_athletes_cleaned.csv', encoding='utf-8-sig')
medal_counts_df = pd.read_csv('summerOly_medal_counts_cleaned.csv', encoding='utf-8-sig')
hosts_df = pd.read_csv('summerOly_hosts_cleaned.csv', encoding='utf-8-sig')
programs_df = pd.read_csv('summerOly_programs_cleaned.csv', encoding='utf-8-sig')

print("\n[Step 1] 构建国家-年份基础表...")

# 获取所有参赛国家和年份的组合
all_years = sorted(athletes_df['Year'].unique())
all_countries = sorted(athletes_df['NOC'].unique())

# 创建国家-年份的完整网格
country_year_df = pd.DataFrame({
    'NOC': np.repeat(all_countries, len(all_years)),
    'Year': np.tile(all_years, len(all_countries))
})

print(f"  ✓ 构建了 {len(country_year_df)} 个(NOC, Year)组合")

# ============== 特征1：历史奖牌数 (Lag Features) ==============
print("\n[Step 2] 添加历史奖牌数特征...")

# 从medal_counts_df中计算各国各年的奖牌数
medal_features = medal_counts_df.groupby(['NOC', 'Year']).agg({
    'Gold': 'sum',
    'Silver': 'sum', 
    'Bronze': 'sum',
    'Total': 'sum'
}).reset_index().rename(columns={
    'Gold': 'Gold_Medals',
    'Silver': 'Silver_Medals',
    'Bronze': 'Bronze_Medals',
    'Total': 'Total_Medals'
})

# 合并到基础表
country_year_df = country_year_df.merge(medal_features, on=['NOC', 'Year'], how='left')

# 填充缺失值为0（表示未赢得奖牌）
country_year_df['Gold_Medals'] = country_year_df['Gold_Medals'].fillna(0).astype(int)
country_year_df['Silver_Medals'] = country_year_df['Silver_Medals'].fillna(0).astype(int)
country_year_df['Bronze_Medals'] = country_year_df['Bronze_Medals'].fillna(0).astype(int)
country_year_df['Total_Medals'] = country_year_df['Total_Medals'].fillna(0).astype(int)

print("  ✓ 添加了当年奖牌数特征")

# 滞后特征（上一届、上上届等）
country_year_df = country_year_df.sort_values(['NOC', 'Year'])

for lag in [1, 2, 3]:
    country_year_df[f'Lag_{lag}_Gold'] = country_year_df.groupby('NOC')['Gold_Medals'].shift(lag)
    country_year_df[f'Lag_{lag}_Total'] = country_year_df.groupby('NOC')['Total_Medals'].shift(lag)

# 滚动平均
country_year_df['Avg_3yr_Gold'] = country_year_df.groupby('NOC')['Gold_Medals'].shift(1).rolling(3, min_periods=1).mean()
country_year_df['Avg_3yr_Total'] = country_year_df.groupby('NOC')['Total_Medals'].shift(1).rolling(3, min_periods=1).mean()

print("  ✓ 添加了滞后和滚动平均特征")

# ============== 特征2：东道主效应 (Host Effect) ==============
print("\n[Step 3] 添加东道主效应特征...")

# 简化方法：直接使用Medal_counts数据中的排名信息
# 根据主办国信息创建映射
host_year_noc = {}
for year in athletes_df['Year'].unique():
    year_athletes = athletes_df[athletes_df['Year'] == year]
    # 找出该年份最常见的"主队"国家（通常是主办方的运动员最多）
    # 这是一个简化的近似
    noc_counts = year_athletes['NOC'].value_counts()
    if len(noc_counts) > 0:
        # 主办方通常派遣相对较多的运动员
        # 但这不准确。更好的方法是使用已知的主办方列表
        pass

# 使用一个手动的主办国映射（基于历史知识）
known_hosts = {
    1896: 'GRE',   # Athens
    1900: 'FRA',   # Paris
    1904: 'USA',   # St. Louis
    1908: 'GBR',   # London
    1912: 'SWE',   # Stockholm
    1920: 'BEL',   # Antwerp
    1924: 'FRA',   # Paris
    1928: 'NED',   # Amsterdam
    1932: 'USA',   # Los Angeles
    1936: 'GER',   # Berlin
    1948: 'GBR',   # London
    1952: 'FIN',   # Helsinki
    1956: 'AUS',   # Melbourne
    1960: 'ITA',   # Rome
    1964: 'JPN',   # Tokyo
    1968: 'MEX',   # Mexico City
    1972: 'GER',   # Munich
    1976: 'CAN',   # Montreal
    1980: 'URS',   # Moscow (removed in cleaning)
    1984: 'USA',   # Los Angeles
    1988: 'KOR',   # Seoul
    1992: 'ESP',   # Barcelona
    1996: 'USA',   # Atlanta
    2000: 'AUS',   # Sydney
    2004: 'GRE',   # Athens
    2008: 'CHN',   # Beijing
    2012: 'GBR',   # London
    2016: 'BRA',   # Rio
    2020: 'JPN',   # Tokyo (2021)
    2024: 'FRA',   # Paris
}

country_year_df['Is_Host'] = country_year_df.apply(
    lambda row: (1 if known_hosts.get(row['Year']) == row['NOC'] else 0),
    axis=1
)

print(f"  ✓ 识别了 {country_year_df['Is_Host'].sum()} 个主办国记录")

# ============== 特征3：运动员投入 (Athlete Power) ==============
print("\n[Step 4] 添加运动员投入特征...")

# 计算各国各年的运动员数和女性比例
athlete_features = athletes_df.groupby(['NOC', 'Year']).agg({
    'Name': 'count',  # 总运动员数（去重后）
    'Sex': lambda x: (x == 'F').sum()  # 女性数量
}).reset_index().rename(columns={
    'Name': 'Athlete_Count',
    'Sex': 'Female_Athletes'
})

athlete_features['Female_Ratio'] = athlete_features['Female_Athletes'] / athlete_features['Athlete_Count']

country_year_df = country_year_df.merge(athlete_features, on=['NOC', 'Year'], how='left')

# 填充缺失值
country_year_df['Athlete_Count'] = country_year_df['Athlete_Count'].fillna(0).astype(int)
country_year_df['Female_Athletes'] = country_year_df['Female_Athletes'].fillna(0).astype(int)
country_year_df['Female_Ratio'] = country_year_df['Female_Ratio'].fillna(0)

print("  ✓ 添加了运动员数量和性别比例特征")

# ============== 特征4：项目覆盖度 (Sport Coverage) ==============
print("\n[Step 5] 添加项目覆盖度特征...")

# 计算各国各年参加的项目数
sport_coverage = athletes_df.groupby(['NOC', 'Year']).agg({
    'Sport': 'nunique',  # 不同运动种类数
    'Event': 'nunique'   # 不同项目数
}).reset_index().rename(columns={
    'Sport': 'Sport_Count',
    'Event': 'Event_Count'
})

country_year_df = country_year_df.merge(sport_coverage, on=['NOC', 'Year'], how='left')
country_year_df['Sport_Count'] = country_year_df['Sport_Count'].fillna(0).astype(int)
country_year_df['Event_Count'] = country_year_df['Event_Count'].fillna(0).astype(int)

print("  ✓ 添加了项目覆盖度特征")

# ============== 特征5：总奥运会金牌数 (Total Olympic Events) ==============
print("\n[Step 6] 添加当年总金牌数特征...")

# 从medal_counts_df计算当年全球总金牌数
total_events_by_year = medal_counts_df.groupby('Year')['Gold'].sum().reset_index().rename(columns={
    'Gold': 'Total_Gold_in_Olympics'
})

country_year_df = country_year_df.merge(total_events_by_year, on='Year', how='left')

print("  ✓ 添加了当年全球总金牌数特征")

# ============== 特征6：项目效率特征 (Sport Efficiency - for Coach Effect Analysis) ==============
print("\n[Step 7] 计算项目效率特征...")

# 计算各国在各项目的奖牌效率（奖牌数/参赛人数）
athletes_with_medals = athletes_df[athletes_df['Medal'] != 'No medal'].copy()

medal_by_country_sport = athletes_with_medals.groupby(['NOC', 'Year', 'Sport']).agg({
    'Medal': 'count'
}).reset_index().rename(columns={'Medal': 'Sport_Medals'})

athlete_by_country_sport = athletes_df.groupby(['NOC', 'Year', 'Sport']).size().reset_index(name='Sport_Athletes')

sport_efficiency = medal_by_country_sport.merge(
    athlete_by_country_sport, 
    on=['NOC', 'Year', 'Sport'], 
    how='left'
)

sport_efficiency['Sport_Efficiency'] = sport_efficiency['Sport_Medals'] / sport_efficiency['Sport_Athletes']

# 计算每个国家每年的平均效率
avg_sport_efficiency = sport_efficiency.groupby(['NOC', 'Year'])['Sport_Efficiency'].mean().reset_index().rename(
    columns={'Sport_Efficiency': 'Avg_Sport_Efficiency'}
)

country_year_df = country_year_df.merge(avg_sport_efficiency, on=['NOC', 'Year'], how='left')
country_year_df['Avg_Sport_Efficiency'] = country_year_df['Avg_Sport_Efficiency'].fillna(0)

print("  ✓ 添加了项目效率特征（用于教练效应分析）")

# ============== 保存特征数据集 ==============
print("\n[Step 8] 保存特征数据集...")

country_year_df = country_year_df.sort_values(['NOC', 'Year']).reset_index(drop=True)
country_year_df.to_csv('country_year_features.csv', index=False, encoding='utf-8')

print("  ✓ summerOly_country_year_features.csv 已保存")

# ============== 生成特征统计报告 ==============
print("\n" + "=" * 80)
print("特征提取完成报告")
print("=" * 80)

print("\n【特征数据集规模】")
print(f"总记录数: {len(country_year_df)}")
print(f"国家数: {country_year_df['NOC'].nunique()}")
print(f"年份数: {country_year_df['Year'].nunique()}")
print(f"特征列数: {len(country_year_df.columns)}")

print("\n【特征列表】")
print("当前目标变量:")
print("  - Gold_Medals: 金牌数")
print("  - Total_Medals: 总奖牌数")
print("\n历史特征（滞后）:")
print("  - Lag_1_Gold, Lag_2_Gold, Lag_3_Gold: 上n届金牌数")
print("  - Lag_1_Total, Lag_2_Total, Lag_3_Total: 上n届总奖牌数")
print("  - Avg_3yr_Gold, Avg_3yr_Total: 过去3年平均奖牌数")
print("\n东道主特征:")
print("  - Is_Host: 是否为主办国 (0/1)")
print("\n运动员特征:")
print("  - Athlete_Count: 当年派出运动员数")
print("  - Female_Ratio: 女性运动员比例")
print("\n项目特征:")
print("  - Sport_Count: 参赛运动种类数")
print("  - Event_Count: 参赛项目总数")
print("  - Total_Gold_in_Olympics: 当年奥运会总金牌数")
print("\n教练效应相关特征:")
print("  - Avg_Sport_Efficiency: 平均项目效率（奖牌数/参赛人数）")

print("\n【缺失值统计】")
missing_summary = country_year_df.isnull().sum()
if missing_summary.sum() > 0:
    print(missing_summary[missing_summary > 0])
else:
    print("  无缺失值 ✓")

print("\n【数据样本示例】")
print(country_year_df.head(10).to_string())

print("\n【统计摘要】")
print(country_year_df[['Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Is_Host', 'Avg_Sport_Efficiency']].describe())

print("\n特征提取完成! ✓")
print("=" * 80)
