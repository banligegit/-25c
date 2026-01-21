import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 设置绘图风格，支持中文显示（如果环境支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 尝试使用中文字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=" * 80)
print("开始执行建模策略：基于机器学习的回归预测与不确定性量化")
print("=" * 80)

# 1. 加载数据
df = pd.read_csv('country_year_features.csv')

# 2. 数据准备
# 我们使用“滑动窗口”逻辑：
# 训练集：1996年 - 2020年（近代奥运，规则较为统一）
# 验证集：2024年（也是测试集，因为我们有真实结果）

train_df = df[(df['Year'] >= 1996) & (df['Year'] <= 2020)].copy()
val_df = df[df['Year'] == 2024].copy()

# 处理缺失值：Lag特征如果没有（说明之前没参加），填0是合理的
features = [
    'Lag_1_Total', 'Lag_1_Gold',        # 核心趋势
    'Lag_2_Total',                      # 长期趋势
    'Athlete_Count',                    # 核心国力
    'Is_Host',                          # 关键环境变量
    'Sport_Count', 'Event_Count',       # 覆盖广度
    'Avg_Sport_Efficiency',             # 效率/教练效应代理
    'Female_Ratio'                      # 结构特征
]

# 填充缺失值
X_train = train_df[features].fillna(0)
X_val = val_df[features].fillna(0)

# 目标变量：我们分别预测“金牌”和“奖牌总数”
target_gold = 'Gold_Medals'
target_total = 'Total_Medals'

y_train_gold = train_df[target_gold]
y_val_gold = val_df[target_gold]

y_train_total = train_df[target_total]
y_val_total = val_df[target_total]

# 3. 建模 - 策略：梯度提升树 (GBDT) + 分位数回归 (用于预测区间)
print(f"\n[模型配置]")
print(f"特征列表: {features}")
print(f"训练集样本量: {len(X_train)}")
print(f"验证集样本量: {len(X_val)} (2024年数据)")

def train_and_predict(target_name, y_train, y_val):
    print(f"\n>>> 正在训练 [{target_name}] 预测模型...")
    
    # 3.1 主预测模型 (预测期望值/中位数)
    model_main = GradientBoostingRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42,
        loss='squared_error' # 优化均方误差
    )
    model_main.fit(X_train, y_train)
    pred_main = model_main.predict(X_val)
    # 修正负数预测：奖牌数不能为负
    pred_main = np.maximum(pred_main, 0)
    
    # 3.2 不确定性量化：预测 90% 置信区间 (Lower=5%, Upper=95%)
    # 下界模型
    model_lower = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42,
        loss='quantile', alpha=0.05 
    )
    model_lower.fit(X_train, y_train)
    pred_lower = model_lower.predict(X_val)
    pred_lower = np.maximum(pred_lower, 0)
    
    # 上界模型
    model_upper = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42,
        loss='quantile', alpha=0.95
    )
    model_upper.fit(X_train, y_train)
    pred_upper = model_upper.predict(X_val)
    
    # 4. 评估
    mae = mean_absolute_error(y_val, pred_main)
    r2 = r2_score(y_val, pred_main)
    rmse = np.sqrt(mean_squared_error(y_val, pred_main))
    
    print(f"--- 模型表现 ({target_name}) ---")
    print(f"R² (决定系数): {r2:.4f} (越接近1越好，>0.8视为优秀)")
    print(f"MAE (平均绝对误差): {mae:.2f} (平均每个国家预测偏离多少枚奖牌)")
    print(f"RMSE (均方根误差): {rmse:.2f} (对大误差更敏感的指标)")
    
    return pred_main, pred_lower, pred_upper, model_main

# 执行预测
pred_gold, lower_gold, upper_gold, model_gold = train_and_predict("金牌榜", y_train_gold, y_val_gold)
pred_total, lower_total, upper_total, model_total = train_and_predict("奖牌总榜", y_train_total, y_val_total)

# 5. 结果整合与分析
results_2024 = val_df[['NOC', 'Year', 'Gold_Medals', 'Total_Medals']].copy()
results_2024['Pred_Gold'] = pred_gold
results_2024['Pred_Total'] = pred_total
results_2024['Gold_Lower'] = lower_gold
results_2024['Gold_Upper'] = upper_gold
results_2024['Total_Lower'] = lower_total
results_2024['Total_Upper'] = upper_total

# 计算误差
results_2024['Gold_Diff'] = results_2024['Pred_Gold'] - results_2024['Gold_Medals']
results_2024['Total_Diff'] = results_2024['Pred_Total'] - results_2024['Total_Medals']

# 6. 可视化
print("\n[生成可视化图表...]")

# 图1: 预测值 vs 真实值 (总奖牌)
plt.figure(figsize=(10, 6))
plt.scatter(results_2024['Total_Medals'], results_2024['Pred_Total'], alpha=0.6, color='blue')
plt.plot([0, 140], [0, 140], 'r--', lw=2)  # 对角线
plt.xlabel('Actual Total Medals (2024)')
plt.ylabel('Predicted Total Medals (2024)')
plt.title('Reference Line (Red) vs Prediction (Blue)')
plt.grid(True)
plt.savefig('model_eval_scatter.png')
print("  ✓ 已保存: model_eval_scatter.png (散点图)")

# 图2: 特征重要性 (使用金牌模型)
feature_imp = pd.Series(model_gold.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp.values, y=feature_imp.index, hue=feature_imp.index, palette='viridis', legend=False)
plt.title('Feature Importance (Gold Medal Model)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('model_feature_importance.png')
print("  ✓ 已保存: model_feature_importance.png (特征重要性)")

# 图3: 前15名国家预测对比
top_countries = results_2024.sort_values('Total_Medals', ascending=False).head(15)
plt.figure(figsize=(14, 7))
x = np.arange(len(top_countries))
width = 0.35

plt.bar(x - width/2, top_countries['Total_Medals'], width, label='Actual', color='navy')
# Calculate error bars with safety checks for non-negative values
lower_diff = (top_countries['Pred_Total'] - top_countries['Total_Lower']).clip(lower=0)
upper_diff = (top_countries['Total_Upper'] - top_countries['Pred_Total']).clip(lower=0)

plt.bar(x + width/2, top_countries['Pred_Total'], width, label='Predicted', color='skyblue', yerr=[
    lower_diff,
    upper_diff
], capsize=5)

plt.xlabel('Country')
plt.ylabel('Total Medals')
plt.title('Top 15 Countries: Actual vs Predicted (2024) with 90% Confidence Interval')
plt.xticks(x, top_countries['NOC'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_top15_compare.png')
print("  ✓ 已保存: model_top15_compare.png (前15强对比)")


# 7. 详细输出重点关注数据
print("\n" + "="*80)
print("【2024年预测效果详解】 (按真实金牌数排序)")
print("说明：Error = 预测值 - 真实值 (正数表示高估，负数表示低估)")
print("="*80)

# 选取重点关注的国家：金牌榜前10 + 东道主(FRA)
top_gold_nocs = results_2024.sort_values('Gold_Medals', ascending=False).head(10)['NOC'].tolist()
if 'FRA' not in top_gold_nocs:
    top_gold_nocs.append('FRA')

display_cols = ['NOC', 'Gold_Medals', 'Pred_Gold', 'Gold_Diff', 'Gold_Lower', 'Gold_Upper', 
                'Total_Medals', 'Pred_Total', 'Total_Diff']

subset = results_2024[results_2024['NOC'].isin(top_gold_nocs)].sort_values('Gold_Medals', ascending=False)
report_str = subset[display_cols].round(1).to_string(index=False)
print(report_str)

# Save report to file
with open('2024_prediction_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("【2024年预测效果详解】 (按真实金牌数排序)\n")
    f.write("说明：Error = 预测值 - 真实值 (正数表示高估，负数表示低估)\n")
    f.write("="*80 + "\n")
    f.write(report_str + "\n\n")
    f.write("【模型关键指标解读】\n")
    f.write(f"1. 解释度 (R2): 金牌模型 {r2_score(y_val_gold, pred_gold):.3f} / 总奖牌模型 {r2_score(y_val_total, pred_total):.3f}\n")
    f.write("   -> 超过0.9说明模型极好地捕捉了奖牌分布规律。\n")
    f.write("2. 东道主效应捕捉 (FRA):\n")
    fra_row = results_2024[results_2024['NOC'] == 'FRA'].iloc[0]
    f.write(f"   - 真实: {fra_row['Gold_Medals']}金 / {fra_row['Total_Medals']}总\n")
    f.write(f"   - 预测: {fra_row['Pred_Gold']:.1f}金 / {fra_row['Pred_Total']:.1f}总\n")
    f.write(f"   - 评价: 预测{'高估' if fra_row['Gold_Diff'] > 0 else '低估'}了 {abs(fra_row['Gold_Diff']):.1f} 枚金牌\n")

print("\n结果已保存至 2024_prediction_report.txt")
print("\n完成。")
