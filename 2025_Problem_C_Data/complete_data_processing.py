import pandas as pd
import numpy as np

print("=" * 80)
print("完整数据清洗与特征提取")
print("=" * 80)

# ============== 步骤1：读取和检查原始数据 ==============
print("\n[Step 1] 读取原始数据...")

athletes_df = pd.read_csv('summerOly_athletes_cleaned.csv', encoding='utf-8-sig')
medal_counts_df = pd.read_csv('summerOly_medal_counts_cleaned.csv', encoding='utf-8-sig')
hosts_df = pd.read_csv('summerOly_hosts_cleaned.csv', encoding='utf-8-sig')
programs_df = pd.read_csv('summerOly_programs_cleaned.csv', encoding='utf-8-sig')

print(f"✓ 运动员数据: {len(athletes_df)} 条")
print(f"✓ 奖牌数据: {len(medal_counts_df)} 条")

# ============== 步骤2：修复奖牌数据中的NOC映射 ==============
print("\n[Step 2] 修复奖牌数据中的国家代码...")

# 从运动员数据中建立国家名称到代码的映射
country_name_to_code = athletes_df[['Team', 'NOC']].drop_duplicates().set_index('Team')['NOC'].to_dict()

# 添加一些特殊情况
special_mappings = {
    'United States': 'USA',
    'Great Britain': 'GBR',
    'Soviet Union': 'URS',
    'Germany': 'GER',
    'East Germany': 'GDR',
    'West Germany': 'FRG',
    'China': 'CHN',
    'Japan': 'JPN',
    'South Korea': 'KOR',
    'North Korea': 'PRK',
    'Russia': 'RUS',
}

country_name_to_code.update(special_mappings)

# 应用映射到medal_counts
def map_country_name_to_code(name):
    if name in country_name_to_code:
        return country_name_to_code[name]
    # 如果是已经是代码，返回原值
    if len(str(name)) == 3 and str(name).isupper():
        return name
    # 如果找不到，尝试模糊匹配
    for key, code in country_name_to_code.items():
        if key.lower() == str(name).lower():
            return code
    return name  # 返回原值

# 对medal_counts中的NOC列应用映射
medal_counts_df['NOC_Code'] = medal_counts_df['NOC'].apply(map_country_name_to_code)

# 检查有多少条记录被成功映射
unmapped = (medal_counts_df['NOC_Code'] == medal_counts_df['NOC']).sum()
print(f"✓ 成功映射了 {len(medal_counts_df) - unmapped} 条奖牌记录")

if unmapped > 0:
    print(f"⚠ 警告: {unmapped} 条记录未被映射（可能需要手动修复）")
    unmapped_countries = medal_counts_df[medal_counts_df['NOC_Code'] == medal_counts_df['NOC']]['NOC'].unique()
    print(f"  未映射的国家: {unmapped_countries[:10]}")

# 使用映射后的代码
medal_counts_df['NOC'] = medal_counts_df['NOC_Code']
medal_counts_df = medal_counts_df.drop('NOC_Code', axis=1)

print(f"✓ 奖牌数据中的NOC已更新")

# ============== 步骤3：构建国家-年份基础表 ==============
print("\n[Step 3] 构建国家-年份基础表...")

all_years = sorted(athletes_df['Year'].unique())
all_countries_athletes = sorted(athletes_df['NOC'].unique())
all_countries_medals = sorted(medal_counts_df['NOC'].unique())

# 合并两个列表
all_countries = sorted(set(all_countries_athletes) | set(all_countries_medals))

# 创建完整的国家-年份网格
country_year_df = pd.DataFrame({
    'NOC': np.repeat(all_countries, len(all_years)),
    'Year': np.tile(all_years, len(all_countries))
})

print(f"✓ 创建了 {len(country_year_df)} 个(NOC, Year)组合")
print(f"✓ 涵盖 {len(all_countries)} 个国家和 {len(all_years)} 个年份")

# ============== 步骤4：添加当年奖牌数 ==============
print("\n[Step 4] 添加当年奖牌数...")

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

country_year_df = country_year_df.merge(medal_features, on=['NOC', 'Year'], how='left')

# 填充缺失值
for col in ['Gold_Medals', 'Silver_Medals', 'Bronze_Medals', 'Total_Medals']:
    country_year_df[col] = country_year_df[col].fillna(0).astype(int)

print(f"✓ 成功合并 {len(medal_features)} 条奖牌记录")

# ============== 步骤5：添加滞后特征 ==============
print("\n[Step 5] 添加历史滞后特征...")

country_year_df = country_year_df.sort_values(['NOC', 'Year']).reset_index(drop=True)

for lag in [1, 2, 3]:
    country_year_df[f'Lag_{lag}_Gold'] = country_year_df.groupby('NOC')['Gold_Medals'].shift(lag)
    country_year_df[f'Lag_{lag}_Total'] = country_year_df.groupby('NOC')['Total_Medals'].shift(lag)

# 滚动平均（使用前面的数据）
country_year_df['Avg_3yr_Gold'] = country_year_df.groupby('NOC')['Gold_Medals'].shift(1).rolling(3, min_periods=1).mean()
country_year_df['Avg_3yr_Total'] = country_year_df.groupby('NOC')['Total_Medals'].shift(1).rolling(3, min_periods=1).mean()

print("✓ 添加了滞后和滚动平均特征")

# ============== 步骤6：添加东道主特征 ==============
print("\n[Step 6] 添加东道主特征...")

known_hosts = {
    1896: 'GRE', 1900: 'FRA', 1904: 'USA', 1908: 'GBR', 1912: 'SWE',
    1920: 'BEL', 1924: 'FRA', 1928: 'NED', 1932: 'USA', 1936: 'GER',
    1948: 'GBR', 1952: 'FIN', 1956: 'AUS', 1960: 'ITA', 1964: 'JPN',
    1968: 'MEX', 1972: 'GER', 1976: 'CAN', 1980: 'URS', 1984: 'USA',
    1988: 'KOR', 1992: 'ESP', 1996: 'USA', 2000: 'AUS', 2004: 'GRE',
    2008: 'CHN', 2012: 'GBR', 2016: 'BRA', 2020: 'JPN', 2024: 'FRA',
}

country_year_df['Is_Host'] = country_year_df.apply(
    lambda row: (1 if known_hosts.get(row['Year']) == row['NOC'] else 0),
    axis=1
)

print(f"✓ 识别了 {country_year_df['Is_Host'].sum()} 个主办国记录")

# ============== 步骤7：添加运动员特征 ==============
print("\n[Step 7] 添加运动员特征...")

athlete_features = athletes_df.groupby(['NOC', 'Year']).agg({
    'Name': 'count',
    'Sex': lambda x: (x == 'F').sum()
}).reset_index().rename(columns={
    'Name': 'Athlete_Count',
    'Sex': 'Female_Athletes'
})

athlete_features['Female_Ratio'] = (athlete_features['Female_Athletes'] / 
                                     athlete_features['Athlete_Count'].clip(lower=1))

country_year_df = country_year_df.merge(athlete_features, on=['NOC', 'Year'], how='left')

for col in ['Athlete_Count', 'Female_Athletes']:
    country_year_df[col] = country_year_df[col].fillna(0).astype(int)

country_year_df['Female_Ratio'] = country_year_df['Female_Ratio'].fillna(0)

print(f"✓ 添加了 {len(athlete_features)} 条运动员特征")

# ============== 步骤8：添加项目特征 ==============
print("\n[Step 8] 添加项目特征...")

sport_coverage = athletes_df.groupby(['NOC', 'Year']).agg({
    'Sport': 'nunique',
    'Event': 'nunique'
}).reset_index().rename(columns={
    'Sport': 'Sport_Count',
    'Event': 'Event_Count'
})

country_year_df = country_year_df.merge(sport_coverage, on=['NOC', 'Year'], how='left')
country_year_df['Sport_Count'] = country_year_df['Sport_Count'].fillna(0).astype(int)
country_year_df['Event_Count'] = country_year_df['Event_Count'].fillna(0).astype(int)

print("✓ 添加了项目覆盖特征")

# ============== 步骤9：添加全局奥运特征 ==============
print("\n[Step 9] 添加全局奥运特征...")

total_medals_by_year = medal_counts_df.groupby('Year')['Gold'].sum().reset_index().rename(
    columns={'Gold': 'Total_Gold_in_Olympics'}
)

country_year_df = country_year_df.merge(total_medals_by_year, on='Year', how='left')
country_year_df['Total_Gold_in_Olympics'] = country_year_df['Total_Gold_in_Olympics'].fillna(0)

print("✓ 添加了全局奥运会金牌数特征")

# ============== 步骤10：添加项目效率特征 ==============
print("\n[Step 10] 计算项目效率特征...")

athletes_with_medals = athletes_df[athletes_df['Medal'] != 'No medal'].copy()
medal_by_sport = athletes_with_medals.groupby(['NOC', 'Year', 'Sport']).size().reset_index(name='Sport_Medals')
athlete_by_sport = athletes_df.groupby(['NOC', 'Year', 'Sport']).size().reset_index(name='Sport_Athletes')

sport_efficiency = medal_by_sport.merge(athlete_by_sport, on=['NOC', 'Year', 'Sport'], how='left')
sport_efficiency['Sport_Efficiency'] = (sport_efficiency['Sport_Medals'] / 
                                         sport_efficiency['Sport_Athletes'].clip(lower=1))

avg_efficiency = sport_efficiency.groupby(['NOC', 'Year'])['Sport_Efficiency'].mean().reset_index().rename(
    columns={'Sport_Efficiency': 'Avg_Sport_Efficiency'}
)

country_year_df = country_year_df.merge(avg_efficiency, on=['NOC', 'Year'], how='left')
country_year_df['Avg_Sport_Efficiency'] = country_year_df['Avg_Sport_Efficiency'].fillna(0)

print("✓ 添加了项目效率特征")

# ============== 步骤11：保存特征数据集 ==============
print("\n[Step 11] 保存特征数据集...")

country_year_df = country_year_df.sort_values(['NOC', 'Year']).reset_index(drop=True)
country_year_df.to_csv('country_year_features.csv', index=False, encoding='utf-8')

print("✓ 已保存: country_year_features.csv")

# ============== 生成报告 ==============
print("\n" + "=" * 80)
print("数据处理完成报告")
print("=" * 80)

print("\n【最终特征数据集】")
print(f"总记录数: {len(country_year_df)}")
print(f"国家数: {country_year_df['NOC'].nunique()}")
print(f"年份范围: {country_year_df['Year'].min()}-{country_year_df['Year'].max()}")
print(f"特征列数: {len(country_year_df.columns)}")

print("\n【奖牌统计】")
gold_total = country_year_df['Gold_Medals'].sum()
total_total = country_year_df['Total_Medals'].sum()
print(f"总金牌数: {gold_total:.0f}")
print(f"总奖牌数: {total_total:.0f}")

print("\n【特征验证 - 2024年中国】")
chn_2024 = country_year_df[(country_year_df['NOC'] == 'CHN') & (country_year_df['Year'] == 2024)]
if len(chn_2024) > 0:
    print(chn_2024[['NOC', 'Year', 'Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Is_Host']].to_string())
else:
    print("未找到2024年中国数据")

print("\n【特征验证 - 2024年美国】")
usa_2024 = country_year_df[(country_year_df['NOC'] == 'USA') & (country_year_df['Year'] == 2024)]
if len(usa_2024) > 0:
    print(usa_2024[['NOC', 'Year', 'Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Is_Host']].to_string())
else:
    print("未找到2024年美国数据")

print("\n【特征验证 - 2024年法国(主办国)】")
fra_2024 = country_year_df[(country_year_df['NOC'] == 'FRA') & (country_year_df['Year'] == 2024)]
if len(fra_2024) > 0:
    print(fra_2024[['NOC', 'Year', 'Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Is_Host']].to_string())
else:
    print("未找到2024年法国数据")

print("\n数据处理完成! ✓")
print("=" * 80)
