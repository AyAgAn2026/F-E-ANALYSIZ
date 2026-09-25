# --- صندوق رفع الملفات في القائمة الجانبية ---
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
    # البيانات الافتراضية للتدريب (في حال عدم رفع ملف)
    np.random.seed(42)
    data = {
        'Fertilizer': np.repeat(['Control', 'NPK_Low', 'NPK_Medium', 'NPK_High'], 3),
        'Block': np.tile(['B1', 'B2', 'B3'], 4),
        'Yield': [12.5, 13.0, 12.8, 15.1, 14.8, 15.5, 18.2, 17.9, 18.5, 22.0, 21.8, 22.5],
        'Plant_Height': [60, 62, 61, 70, 72, 71, 85, 83, 86, 95, 94, 96]
    }
    df = pd.DataFrame(data)

    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x='Plant_Height', y='Yield', data=df, ax=ax, color="darkgreen")
    ax.set_title("Regression Line")
    st.pyplot(fig)
