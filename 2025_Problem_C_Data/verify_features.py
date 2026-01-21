import pandas as pd
import numpy as np

print("\n" + "="*80)
print("特征数据集验证报告")
print("="*80)

df = pd.read_csv('country_year_features.csv')

print(f"\n【基础信息】")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"数据类型: {df.shape}")

print(f"\n【列名】")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n【数据范围】")
print(f"国家数: {df['NOC'].nunique()}")
print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"年份数: {df['Year'].nunique()}")

print(f"\n【2024年参赛国（前15名）】")
df_2024 = df[df['Year'] == 2024].copy()
print(f"2024年参赛国总数: {len(df_2024[df_2024['Athlete_Count'] > 0])}")
print(f"2024年获得奖牌的国家: {len(df_2024[df_2024['Total_Medals'] > 0])}")

top_15 = df_2024.nlargest(15, 'Gold_Medals')[['NOC', 'Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Is_Host']]
print(top_15.to_string(index=False))

print(f"\n【2024年关键国家检验】")
for country in ['CHN', 'USA', 'JPN', 'AUS', 'FRA']:
    row = df_2024[df_2024['NOC'] == country]
    if len(row) > 0:
        print(f"{country}: {row['Gold_Medals'].values[0]:.0f}金 {row['Total_Medals'].values[0]:.0f}总 "
              f"({row['Athlete_Count'].values[0]:.0f}名运动员) Host={row['Is_Host'].values[0]:.0f}")

print(f"\n【特征缺失值统计】")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("有缺失值的列:")
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
else:
    print("  无缺失值 ✓")

print(f"\n【数值特征统计】")
numeric_cols = ['Gold_Medals', 'Total_Medals', 'Athlete_Count', 'Avg_3yr_Gold', 'Avg_Sport_Efficiency']
stats = df[numeric_cols].describe()
print(stats)

print(f"\n【分类特征统计】")
print(f"Is_Host: {df['Is_Host'].value_counts().to_dict()}")

print(f"\n【历史数据示例 - 中国（CHN）】")
china = df[df['NOC'] == 'CHN'].sort_values('Year')
print(china[['Year', 'Gold_Medals', 'Total_Medals', 'Athlete_Count']].tail(8).to_string(index=False))

print(f"\n【历史数据示例 - 美国（USA）】")
usa = df[df['NOC'] == 'USA'].sort_values('Year')
print(usa[['Year', 'Gold_Medals', 'Total_Medals', 'Athlete_Count']].tail(8).to_string(index=False))

print(f"\n【东道主效应分析】")
host_years = df[df['Is_Host'] == 1][['Year', 'NOC', 'Gold_Medals', 'Total_Medals']].sort_values('Year')
if len(host_years) > 0:
    print(f"识别的主办国（共{len(host_years)}条）:")
    print(host_years.to_string(index=False))
    
    # 比较主办国与非主办国的平均奖牌数
    hosts_avg = df[df['Is_Host'] == 1]['Total_Medals'].mean()
    non_hosts_avg = df[df['Is_Host'] == 0]['Total_Medals'].mean()
    print(f"\n主办国平均奖牌数: {hosts_avg:.2f}")
    print(f"非主办国平均奖牌数: {non_hosts_avg:.2f}")
    print(f"东道主溢价: {(hosts_avg/non_hosts_avg - 1)*100:.1f}%")

print(f"\n【特征相关性（与Total_Medals）】")
correlations = df[['Gold_Medals', 'Lag_1_Gold', 'Lag_1_Total', 'Athlete_Count', 'Sport_Count', 'Total_Medals']].corr()['Total_Medals'].sort_values(ascending=False)
print(correlations[correlations.index != 'Total_Medals'])

print(f"\n验证完成! ✓")
print("="*80 + "\n")
