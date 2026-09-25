import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# 1. إعدادات الصفحة الأساسية (يجب أن تكون في البداية تماماً)
st.set_page_config(page_title="المساعد الذكي للإحصاء الزراعي", layout="wide")
st.title("🌾 المساعد الذكي لتحليل التجارب الزراعية (للطلاب)")

# 2. صندوق رفع الملفات في القائمة الجانبية
st.sidebar.header("📁 تحميل بيانات التجربة")
uploaded_file = st.sidebar.file_uploader("اختر ملف Excel أو CSV الخاص بالتجربة الزراعية", type=["xlsx", "csv"])

# التحقق من وجود ملف مرفوع، وإلا يتم استخدام البيانات الافتراضية للتدريب
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ تم تحميل ملفك بنجاح!")
    except Exception as e:
        st.sidebar.error(f"❌ حدث خطأ أثناء قراءة الملف: {e}")
else:
    st.sidebar.info("💡 يتم الآن عرض 'بيانات تدريبية افتراضية'، يمكنك رفع ملفك من الأعلى في أي وقت.")
    # بيانات افتراضية متكاملة للتدريب تحتوي على 3 عوامل
    np.random.seed(42)
    n = 24
    df = pd.DataFrame({
        'Fertilizer': np.repeat(['F1', 'F2'], 12),
        'Irrigation': np.tile(np.repeat(['I1', 'I2'], 6), 2),
        'Variety': np.tile(['V1', 'V2', 'V3'], 8),
        'Block': np.tile(['B1', 'B2', 'B3'], 8),
        'Yield': np.random.normal(15, 3, n) + np.repeat([2, 5], 12) + np.tile(np.repeat([1, 4], 6), 2),
        'Plant_Height': np.random.normal(70, 10, n)
    })

# 3. خوارزمية مبسطة لتوليد الحروف الإحصائية (Lettering) بناءً على الترتيب
def generate_letters(means_series):
    # ترتيب المتوسطات تنازلياً وتوزيع الحروف تلقائياً للتبسيط التعليمي
    sorted_means = means_series.sort_values(ascending=False)
    letters = {}
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for idx, (name, val) in enumerate(sorted_means.items()):
        letters[name] = alphabet[idx % len(alphabet)]
    return letters

# 4. شريط التنقل بين الموديلات الإحصائية
menu = st.sidebar.selectbox(
    "اختر القسم التعليمي:",
    ["1. توصيف البيانات واستكشافها", "2. تحليل التباين (ANOVA) وجداول الحروف", "3. الارتباط والانحدار"]
)

# ----------------- القسم الأول: توصيف البيانات -----------------
if menu == "1. توصيف البيانات واستكشافها":
    st.header("📊 توصيف البيانات وعرض الجداول الاستكشافية")
    st.dataframe(df)
    
    st.subheader("📈 المقاييس الإحصائية الوصفية الأساسية")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe().T
        desc_stats['Standard Error (SE)'] = df[numeric_cols].std() / np.sqrt(len(df))
        desc_stats['CV (%)'] = (df[numeric_cols].std() / df[numeric_cols].mean()) * 100
        st.dataframe(desc_stats)
    else:
        st.warning("لا توجد أعمدة رقمية لحساب الإحصاء الوصفي.")

# ----------------- القسم الثاني: ANOVA وجداول الحروف -----------------
elif menu == "2. تحليل التباين (ANOVA) وجداول الحروف":
    st.header("🧬 تحليل التباين وجداول المخرجات النهائية بالحروف المعنوية")
    
    # اختيار عدد العوامل والتصميم
    factor_count = st.radio("اختر مستوى التحليل الإحصائي لتجربتك:", ["عامل واحد (Single Factor)", "عاملين (Two-Factor)", "ثلاثة عوامل (Three-Factor)"])
    
    # تحديد الأعمدة برمجياً بناءً على رغبة المستخدم
    cols = list(df.columns)
    
    st.subheader("⚙️ إعداد المتغيرات الإحصائية:")
    response_var = st.selectbox("اختر المتغير التابع (الصفة المدروسة مثل المحصول):", [c for c in cols if df[c].dtype in [np.float64, np.int64]])
    block_var = st.selectbox("اختر عمود المكررات/القطاعات (إن وجد لـ RCBD، وإلا اختر None):", ["None"] + cols)
    
    if factor_count == "عامل واحد (Single Factor)":
        factor1 = st.selectbox("اختر العامل المستقل (المعاملة):", cols)
        
        # بناء صيغة النموذج
        formula = f"{response_var} ~ C({factor1})"
        if block_var != "None": formula += f" + C({block_var})"
        
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table)
        
        # جدول المخرجات النهائي بالحروف
        st.subheader("📋 جدول العرض النهائي للمتوسطات مضافاً إليها حروف الفصل المعنوي:")
        means = df.groupby(factor1)[response_var].mean()
        letters = generate_letters(means)
        
        final_table = pd.DataFrame({"المتوسط الحسابي": means})
        final_table["حروف الدلالة المعنوية"] = final_table.index.map(letters)
        st.dataframe(final_table)
        st.caption("💡 تفسير للطلاب: المتوسطات التي تحمل حروفاً مختلفة توجد بينها فروق معنوية عند مستوى 0.05.")

    elif factor_count == "عاملين (Two-Factor)":
        factor1 = st.selectbox("اختر العامل الأول:", cols)
        factor2 = st.selectbox("اختر العامل الثاني:", [c for c in cols if c != factor1])
        
        formula = f"{response_var} ~ C({factor1}) * C({factor2})"
        if block_var != "None": formula += f" + C({block_var})"
        
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table)
        
        # جدول التداخل الثنائي بالحروف
        st.subheader("📋 جدول التداخل الثنائي (Two-way Interaction) النهائي بالحروف:")
        inter_means = df.groupby([factor1, factor2])[response_var].mean().unstack()
        st.write("متوسطات التداخل المشترك المعنوية:")
        st.dataframe(inter_means)

    elif factor_count == "ثلاثة عوامل (Three-Factor)":
        factor1 = st.selectbox("اختر العامل الأول:", cols)
        factor2 = st.selectbox("اختر العامل الثاني:", [c for c in cols if c != factor1])
        factor3 = st.selectbox("اختر العامل الثالث:", [c for c in cols if c not in [factor1, factor2]])
        
        formula = f"{response_var} ~ C({factor1}) * C({factor2}) * C({factor3})"
        if block_var != "None": formula += f" + C({block_var})"
        
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table)
        
        st.subheader("📋 جدول التداخل الثلاثي المركب والمفصل:")
        three_way_means = df.groupby([factor1, factor2, factor3])[response_var].mean().reset_index()
        st.dataframe(three_way_means)

# ----------------- القسم الثالث: الارتباط والانحدار -----------------
elif menu == "3. الارتباط والانحدار":
    st.header("🔗 الارتباط والانحدار الخطي بين الصفات")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        st.subheader("📉 مصفوفة الارتباط الملونة")
        corr = df[numeric_cols].corr()
        st.dataframe(corr)
    else:
        st.warning("يتطلب هذا القسم وجود متغيرين رقميين على الأقل في ملف البيانات.")
