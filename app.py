import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

st.set_page_config(page_title="المساعد الذكي للإحصاء الزراعي", layout="wide")

st.markdown("""
<style>
    th, td { text-align: right !important; }
    div[data-testid="stMarkdownContainer"] { text-align: right; direction: RTL; }
    p, h1, h2, h3, h4, h5, h6 { text-align: right; direction: RTL; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 التطبيق التعليمي الذكي لتحليل التجارب الزراعية")
st.write("أداة تعليمية شاملة للطلاب لتحليل البيانات، وتوصيفها، وإجراء الـ ANOVA وحساب حروف الدلالة المعنوية لجميع مستويات العوامل.")

st.sidebar.header("⚙️ لوحة التحكم بالتطبيق")
menu = st.sidebar.selectbox(
    "اختر الموديل الإحصائي:",
    ["1. توصيف البيانات واستكشافها", "2. تحليل التباين (ANOVA) وجداول الحروف", "3. الارتباط والانحدار الخطي"]
)

@st.cache_data
def generate_agricultural_data(num_factors):
    np.random.seed(42)
    blocks = ['B1', 'B2', 'B3']
    
    if num_factors == 1:
        f1 = ['Control', 'NPK_50', 'NPK_100', 'Bio_Fert']
        combos = list(itertools.product(f1, blocks))
        df = pd.DataFrame(combos, columns=['Fertilizer', 'Block'])
        df['Yield'] = [10.5, 11.2, 9.8, 14.2, 13.8, 14.5, 19.1, 18.7, 19.5, 16.2, 15.8, 16.5]
        df['Plant_Height'] = [60, 62, 59, 70, 72, 71, 85, 83, 86, 75, 77, 76]
        return df
    elif num_factors == 2:
        f1 = ['Control', 'NPK_100']
        f2 = ['Variety_A', 'Variety_B']
        combos = list(itertools.product(f1, f2, blocks))
        df = pd.DataFrame(combos, columns=['Fertilizer', 'Variety', 'Block'])
        df['Yield'] = [10.2, 10.8, 9.7, 12.5, 12.1, 13.0, 18.5, 17.9, 19.1, 22.4, 21.8, 23.0]
        df['Plant_Height'] = [60, 61, 58, 65, 67, 66, 82, 80, 84, 90, 89, 92]
        return df
    else:
        f1 = ['Control', 'NPK_100']
        f2 = ['Variety_A', 'Variety_B']
        f3 = ['Irrigation_Low', 'Irrigation_High']
        combos = list(itertools.product(f1, f2, f3, blocks))
        df = pd.DataFrame(combos, columns=['Fertilizer', 'Variety', 'Irrigation', 'Block'])
        base_yields = [10, 12, 13, 15, 17, 19, 21, 25]
        yields = []
        for i, b in enumerate(blocks):
            for y in base_yields:
                yields.append(y + np.random.normal(0, 0.4))
        df['Yield'] = yields
        df['Plant_Height'] = [x * 4 + np.random.normal(0, 2) for x in yields]
        return df

def assign_letters(means_dict, lsd_val):
    sorted_means = sorted(means_dict.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_means)
    letter_chars = "abcdefghijklmnopqrstuvwxyz"
    
    groups = []
    for i in range(n):
        group = [sorted_means[i][0]]
        for j in range(i + 1, n):
            if abs(sorted_means[i][1] - sorted_means[j][1]) <= lsd_val:
                group.append(sorted_means[j][0])
        groups.append(group)
        
    assigned_letters = {item[0]: [] for item in sorted_means}
    for idx, g in enumerate(groups[:4]):
        char = letter_chars[idx] if idx < len(letter_chars) else f"z"
        for item in g:
            if char not in assigned_letters[item]:
                assigned_letters[item].append(char)
                
    final_letters = {k: "".join(sorted(v)) for k, v in assigned_letters.items()}
    for k in final_letters:
        if not final_letters[k]:
            final_letters[k] = "a"
    return final_letters

if menu == "1. توصيف البيانات واستكشافها":
    st.header("📊 موديول توصيف البيانات واستكشافها")
    st.write("يتعلم الطالب هنا كيفية تلخيص الصفات الحقلية وحساب معاملات الاختلاف وتوزيع البيانات.")
    
    num_f = st.radio("اختر عدد العوامل في التجربة لتوليد بيانات التدريب المخصصة:", [1, 2, 3], horizontal=True)
    df = generate_agricultural_data(num_f)
    
    st.subheader("📋 جدول البيانات التجريبية الجاهزة")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("📈 مقاييس الإحصاء الوصفي")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    desc = df[numeric_cols].describe().T
    desc['Standard Error (SE)'] = df[numeric_cols].std() / np.sqrt(len(df))
    desc['CV (%)'] = (df[numeric_cols].std() / df[numeric_cols].mean()) * 100
    st.dataframe(desc[['mean', 'std', 'Standard Error (SE)', 'CV (%)', 'min', 'max']], use_container_width=True)
    
    st.subheader("🔍 استكشاف توزيع البيانات هندسياً")
    target_var = st.selectbox("اختر الصفة المراد رسمها:", numeric_cols)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[target_var], kde=True, ax=ax[0], color="skyblue")
    ax[0].set_title(f"Distribution of {target_var}")
    
    factor_col = df.columns[0]
    sns.boxplot(x=df[factor_col], y=df[target_var], ax=ax[1], palette="Set3")
    ax[1].set_title(f"Boxplot of {target_var} by {factor_col}")
    st.pyplot(fig)

elif menu == "2. تحليل التباين (ANOVA) وجداول الحروف":
    st.header("🧬 موديول تحليل التباين (ANOVA) وتوليد جداول الحروف")
    st.write("يعد هذا الموديول الأهم لإنتاج جداول عرض النتائج النهائية مضافا لها حروف الفصل المعنوي للبحث العلمي.")
    
    num_f = st.radio("اختر عدد العوامل للتجربة الحقلية:", [1, 2, 3], horizontal=True)
    df = generate_agricultural_data(num_f)
    
    st.subheader("⚙️ بناء النموذج الإحصائي")
    
    if num_f == 1:
        formula = "Yield ~ C(Fertilizer) + C(Block)"
        factors = ['Fertilizer']
    elif num_f == 2:
        formula = "Yield ~ C(Fertilizer) * C(Variety) + C(Block)"
        factors = ['Fertilizer', 'Variety', 'Fertilizer:Variety']
    else:
        formula = "Yield ~ C(Fertilizer) * C(Variety) * C(Irrigation) + C(Block)"
        factors = ['Fertilizer', 'Variety', 'Irrigation', 'Fertilizer:Variety', 'Fertilizer:Irrigation', 'Variety:Irrigation', 'Fertilizer:Variety:Irrigation']
        
    st.code(f"النموذج الرياضي المستخدم (ANOVA Formula): {formula}", language="python")
    
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.dataframe(anova_table, use_container_width=True)
    
    st.subheader("💡 التفسير التعليمي لجدول ANOVA:")
    for factor in factors:
        if factor in anova_table.index:
            p_val = anova_table.loc[factor, 'PR(>F)']
            if p_val < 0.05:
                st.success(f"✔️ التأثير لـ ({factor}) **معنوي إحصائياً** لأن قيمة P-value ({p_val:.4f}) أقل من 0.05.")
            else:
                st.warning(f"❌ التأثير لـ ({factor}) **غير معنوي** لأن قيمة P-value ({p_val:.4f}) أكبر من 0.05.")
                
    st.header("📋 جداول العرض النهائية بحروف الدلالة المعنوية")
    st.write("تقوم الخوارزمية بحساب الفروق ومقارنتها بقيمة LSD وتوزيع الحروف تلقائياً:")
    
    mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    lsd = 1.96 * np.sqrt(2 * mse / 3)
    st.metric("قيمة الأقل فرق معنوي المقدرة (LSD value)", f"{lsd:.3f}")
    
    if num_f == 1:
        st.subheader("📊 جدول متوسطات العامل الواحد (Single Factor Table)")
        means = df.groupby('Fertilizer')['Yield'].mean().to_dict()
        letters = assign_letters(means, lsd)
        
        final_df = pd.DataFrame({
            'المتوسط الحسابي (Mean)': [f"{means[k]:.2f} {letters[k]}" for k in means],
        }, index=means.keys())
        st.dataframe(final_df, use_container_width=True)
        
    elif num_f == 2:
        st.subheader("📊 جدول التداخل الثنائي (Two-way Interaction Table)")
        means = df.groupby(['Fertilizer', 'Variety'])['Yield'].mean().to_dict()
        flat_means = {f"{k[0]} x {k[1]}": v for k, v in means.items()}
        letters = assign_letters(flat_means, lsd)
        
        records = []
        for f1 in df['Fertilizer'].unique():
            row = {'السماد / المعاملة': f1}
            for f2 in df['Variety'].unique():
                key = f"{f1} x {f2}"
                row[f2] = f"{means[(f1, f2)]:.2f} {letters[key]}"
            records.append(row)
        
        st.dataframe(pd.DataFrame(records).set_index('السماد / المعاملة'), use_container_width=True)
        
    elif num_f == 3:
        st.subheader("📊 جدول التداخل الثلاثي المركب (Three-way Interaction Table)")
        means = df.groupby(['Fertilizer', 'Variety', 'Irrigation'])['Yield'].mean().to_dict()
        flat_means = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in means.items()}
        letters = assign_letters(flat_means, lsd)
        
        records = []
        for f1 in df['Fertilizer'].unique():
            for f2 in df['Variety'].unique():
                row = {'عامل 1: السماد': f1, 'عامل 2: الصنف': f2}
                for f3 in df['Irrigation'].unique():
                    key = f"{f1}_{f2}_{f3}"
                    row[f3] = f"{means[(f1, f2, f3)]:.2f} {letters[key]}"
                records.append(row)
        st.dataframe(pd.DataFrame(records).set_index(['عامل 1: السماد', 'عامل 2: الصنف']), use_container_width=True)

elif menu == "3. الارتباط والانحدار الخطي":
    st.header("🔗 موديول الارتباط وتحليل الانحدار")
    st.write("يتعلم الطلاب هنا كيفية قياس قوة العلاقة بين صفتين زراعيتين والتنبؤ بإحداهما.")
    
    df = generate_agricultural_data(1)
    
    st.subheader("📉 مصفوفة الارتباط لبيرسون (Pearson Correlation)")
    corr = df[['Yield', 'Plant_Height']].corr()
    st.dataframe(corr)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    
    st.subheader("📈 تحليل الانحدار الخطي البسيط (Simple Linear Regression)")
    st.write("التنبؤ بصفة الإنتاجية بدلالة طول النبات.")
    
    X = sm.add_constant(df['Plant_Height'])
    y = df['Yield']
    reg_model = sm.OLS(y, X).fit()
    
    st.text(reg_model.summary().as_text())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x='Plant_Height', y='Yield', data=df, ax=ax, color="darkgreen")
    ax.set_title("Regression Line")
    st.pyplot(fig)
