import pandas as pd
import numpy as np

# 读取原始数据（处理编码问题）
athletes_df = pd.read_csv('summerOly_athletes.csv', encoding='latin-1')
medal_counts_df = pd.read_csv('summerOly_medal_counts.csv', encoding='latin-1')
programs_df = pd.read_csv('summerOly_programs.csv', encoding='latin-1')
hosts_df = pd.read_csv('summerOly_hosts.csv', encoding='latin-1')

print("=" * 80)
print("开始数据清洗")
print("=" * 80)

# ============== 第1步：处理运动员数据 ==============
print("\n[Step 1] 处理运动员数据...")

# 1.1 澳大拉西亚（ANZ）数据均分到AUS和NZL
anz_1908 = athletes_df[(athletes_df['NOC'] == 'ANZ') & (athletes_df['Year'] == 1908)].copy()
anz_1912 = athletes_df[(athletes_df['NOC'] == 'ANZ') & (athletes_df['Year'] == 1912)].copy()

if len(anz_1908) > 0:
    anz_1908_aus = anz_1908.copy()
    anz_1908_nzl = anz_1908.copy()
    anz_1908_aus['NOC'] = 'AUS'
    anz_1908_nzl['NOC'] = 'NZL'
    athletes_df = pd.concat([athletes_df, anz_1908_aus, anz_1908_nzl], ignore_index=True)
    print(f"  ✓ ANZ 1908: 添加了 {len(anz_1908_aus)} 条AUS记录，{len(anz_1908_nzl)} 条NZL记录")

if len(anz_1912) > 0:
    anz_1912_aus = anz_1912.copy()
    anz_1912_nzl = anz_1912.copy()
    anz_1912_aus['NOC'] = 'AUS'
    anz_1912_nzl['NOC'] = 'NZL'
    athletes_df = pd.concat([athletes_df, anz_1912_aus, anz_1912_nzl], ignore_index=True)
    print(f"  ✓ ANZ 1912: 添加了 {len(anz_1912_aus)} 条AUS记录，{len(anz_1912_nzl)} 条NZL记录")

# 1.2 删除ANZ原始数据
athletes_df = athletes_df[athletes_df['NOC'] != 'ANZ'].reset_index(drop=True)

# 1.3 国家代码映射
# SAA → YEM
athletes_df['NOC'] = athletes_df['NOC'].replace('SAA', 'YEM')
print("  ✓ SAA → YEM")

# VNM → VIE
athletes_df['NOC'] = athletes_df['NOC'].replace('VNM', 'VIE')
print("  ✓ VNM → VIE")

# YAR, YMD → YEM (1990年后)
athletes_df.loc[(athletes_df['NOC'] == 'YAR') & (athletes_df['Year'] >= 1990), 'NOC'] = 'YEM'
athletes_df.loc[(athletes_df['NOC'] == 'YMD') & (athletes_df['Year'] >= 1990), 'NOC'] = 'YEM'
print("  ✓ YAR, YMD (1990年后) → YEM")

# FRG, GDR → GER (1990年后)
athletes_df.loc[(athletes_df['NOC'] == 'FRG') & (athletes_df['Year'] >= 1990), 'NOC'] = 'GER'
athletes_df.loc[(athletes_df['NOC'] == 'GDR') & (athletes_df['Year'] >= 1990), 'NOC'] = 'GER'
print("  ✓ FRG, GDR (1990年后) → GER")

# 1.4 删除不讨论的国家
countries_to_drop = ['AHO', 'BLR', 'BOH', 'CRT', 'EUN', 'IOA', 'LIB', 'MAL', 'NBO', 
                     'NFL', 'RHO', 'ROC', 'RUS', 'UNK', 'URS', 'WIF', 'YUG']

initial_count = len(athletes_df)
athletes_df = athletes_df[~athletes_df['NOC'].isin(countries_to_drop)].reset_index(drop=True)
dropped_count = initial_count - len(athletes_df)

print(f"  ✓ 删除了 {dropped_count} 条不讨论的国家数据")
print(f"    删除的国家: {', '.join(countries_to_drop)}")

# 1.5 处理SCG分割（2006年后）
# SCG解体为SRB和MNE，需要在2006年后将SCG分割
scg_before_2006 = athletes_df[(athletes_df['NOC'] == 'SCG') & (athletes_df['Year'] < 2006)]
if len(scg_before_2006) > 0:
    # SCG 2006年前保留为SCG（虽然后续会删除）
    # 或者根据历史，保留为独立国家数据
    # 这里的策略：保留SCG在2006年前的数据作为历史记录，但在分析中作为独立实体
    print("  ℹ SCG 数据在2006年前保留（尽管2006年后已解体）")

# 1.6 处理TCH分割（1993年后）
tchs_before_1993 = athletes_df[(athletes_df['NOC'] == 'TCH') & (athletes_df['Year'] < 1993)]
if len(tchs_before_1993) > 0:
    print("  ℹ TCH 数据在1993年前保留（尽管1993年后已解体）")

# 1.7 处理UAR分割
uars = athletes_df[athletes_df['NOC'] == 'UAR']
if len(uars) > 0:
    print("  ℹ UAR 数据保留（虽然后续以EGY/SYR身份参赛）")

print(f"\n  运动员数据清洗完成: {len(athletes_df)} 条记录")

# ============== 第2步：处理奖牌统计数据 ==============
print("\n[Step 2] 处理奖牌统计数据...")

# 应用同样的清洗规则到medal_counts_df
countries_to_drop_medal = ['AHO', 'BLR', 'BOH', 'CRT', 'EUN', 'IOA', 'LIB', 'MAL', 'NBO', 
                           'NFL', 'RHO', 'ROC', 'RUS', 'UNK', 'URS', 'WIF', 'YUG']

initial_count = len(medal_counts_df)
medal_counts_df = medal_counts_df[~medal_counts_df['NOC'].isin(countries_to_drop_medal)].reset_index(drop=True)
dropped_count = initial_count - len(medal_counts_df)

print(f"  ✓ 删除了 {dropped_count} 条不讨论的国家奖牌数据")

# 国家代码映射
medal_counts_df['NOC'] = medal_counts_df['NOC'].replace('SAA', 'YEM')
medal_counts_df['NOC'] = medal_counts_df['NOC'].replace('VNM', 'VIE')

medal_counts_df.loc[(medal_counts_df['NOC'] == 'YAR') & (medal_counts_df['Year'] >= 1990), 'NOC'] = 'YEM'
medal_counts_df.loc[(medal_counts_df['NOC'] == 'YMD') & (medal_counts_df['Year'] >= 1990), 'NOC'] = 'YEM'

medal_counts_df.loc[(medal_counts_df['NOC'] == 'FRG') & (medal_counts_df['Year'] >= 1990), 'NOC'] = 'GER'
medal_counts_df.loc[(medal_counts_df['NOC'] == 'GDR') & (medal_counts_df['Year'] >= 1990), 'NOC'] = 'GER'

print(f"  奖牌统计数据清洗完成: {len(medal_counts_df)} 条记录")

# ============== 第3步：处理赛事数据 ==============
print("\n[Step 3] 处理赛事数据...")
# programs_df 中的NOC信息通过hosts_df关联，暂时不需要直接清洗
print(f"  赛事数据保持原状: {len(programs_df)} 条记录")

# ============== 第4步：处理主办国数据 ==============
print("\n[Step 4] 处理主办国数据...")
print(f"  主办国数据保持原状: {len(hosts_df)} 条记录")

# ============== 第5步：去重处理 ==============
print("\n[Step 5] 处理团体项目去重...")

# 对于团体项目，同一国家同一项目同一奖牌只算一次
# 去重的key: Year, NOC, Event, Medal
athletes_dedup = athletes_df.drop_duplicates(subset=['Year', 'NOC', 'Event', 'Medal'], keep='first').reset_index(drop=True)
dedup_count = len(athletes_df) - len(athletes_dedup)

print(f"  ✓ 删除了 {dedup_count} 条重复的团体项目记录")
print(f"  运动员数据去重后: {len(athletes_dedup)} 条记录")

# ============== 保存清洗后的数据 ==============
print("\n[Step 6] 保存清洗后的数据...")

athletes_dedup.to_csv('summerOly_athletes_cleaned.csv', index=False, encoding='utf-8')
medal_counts_df.to_csv('summerOly_medal_counts_cleaned.csv', index=False, encoding='utf-8')
hosts_df.to_csv('summerOly_hosts_cleaned.csv', index=False, encoding='utf-8')
programs_df.to_csv('summerOly_programs_cleaned.csv', index=False, encoding='utf-8')

print("  ✓ summerOly_athletes_cleaned.csv")
print("  ✓ summerOly_medal_counts_cleaned.csv")
print("  ✓ summerOly_hosts_cleaned.csv")
print("  ✓ summerOly_programs_cleaned.csv")

# ============== 生成清洗报告 ==============
print("\n" + "=" * 80)
print("数据清洗报告")
print("=" * 80)

print("\n【原始数据统计】")
print(f"运动员数据: {len(athletes_df)} 条记录")
print(f"奖牌统计数据: {len(medal_counts_df)} 条记录")
print(f"赛事数据: {len(programs_df)} 条记录")
print(f"主办国数据: {len(hosts_df)} 条记录")

print("\n【清洗操作汇总】")
print("✓ 澳大拉西亚(ANZ)数据均分到AUS和NZL")
print("✓ 国家代码映射:")
print("  - SAA → YEM")
print("  - VNM → VIE")
print("  - YAR/YMD (1990年后) → YEM")
print("  - FRG/GDR (1990年后) → GER")
print("✓ 删除了不讨论的国家数据(17个国家)")
print("✓ 去除了团体项目的重复计数")

print("\n【参赛国家总数】")
unique_countries = athletes_dedup['NOC'].unique()
print(f"总共 {len(unique_countries)} 个国家/地区参赛")
print(f"参赛国家代码: {sorted(unique_countries)}")

print("\n【年份覆盖范围】")
years = sorted(athletes_dedup['Year'].unique())
print(f"奥运会年份范围: {years[0]} - {years[-1]} ({len(years)}届)")

print("\n数据清洗完成! ✓")
print("=" * 80)
