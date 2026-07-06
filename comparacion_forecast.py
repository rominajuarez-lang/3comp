import warnings
import pandas as pd
import plotly.express as px
import streamlit as st

from datos import convertir_a_mensual
from generar_pronosticos import generar_forecast_mejor_por_producto

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Comparación Forecast",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Comparación de Forecast")
st.caption("Compara ventas reales, forecast comercial y forecast propuesto por SKU.")


def normalizar_product_id(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def leer_demanda(xls: pd.ExcelFile) -> pd.DataFrame:
    hoja = "Demanda" if "Demanda" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=hoja)
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "día": "date",
        "dia": "date",
        "producto": "product_id",
        "sku": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "grupo de demanda": "product_id",
        "demanda": "demand_real",
        "venta": "demand_real",
        "ventas": "demand_real",
        "cantidad": "demand_real",
        "unidades": "demand_real",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    requeridas = ["date", "product_id", "demand_real"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en Demanda: {faltantes}")

    df = df[requeridas].copy()
    df["product_id"] = normalizar_product_id(df["product_id"])
    return convertir_a_mensual(df)


def leer_forecast_comercial(xls: pd.ExcelFile) -> pd.DataFrame:
    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "period": "date",
        "date": "date",
        "producto": "product_id",
        "sku": "product_id",
        "product_id": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "grupo de demanda": "product_id",
        "grupo_demanda": "product_id",
        "forecast": "forecast_company",
        "forecast comercial": "forecast_company",
        "forecast_comercial": "forecast_company",
        "forecast empresa": "forecast_company",
        "forecast_empresa": "forecast_company",
        "forecast_company": "forecast_company",
        "pronostico": "forecast_company",
        "pronóstico": "forecast_company",
        "pronostico comercial": "forecast_company",
        "pronóstico comercial": "forecast_company",
        "pronostico_comercial": "forecast_company",
        "pronostico_empresa": "forecast_company",
        "pronóstico_empresa": "forecast_company",
    }

    hojas_prioritarias = [
        "Forecast_Comercial",
        "Forecast Comercial",
        "forecast_comercial",
        "forecast comercial",
        "Forescast_Comercial",
        "Forescast Comercial",
        "Pronostico_Comercial",
        "Pronóstico_Comercial",
        "Pronostico Comercial",
        "Pronóstico Comercial",
    ]

    hojas = [h for h in hojas_prioritarias if h in xls.sheet_names]
    hojas += [h for h in xls.sheet_names if h not in hojas]

    for hoja in hojas:
        df = pd.read_excel(xls, sheet_name=hoja)
        if df.empty:
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

        if not all(c in df.columns for c in ["date", "product_id", "forecast_company"]):
            continue

        df = df[["date", "product_id", "forecast_company"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["product_id"] = normalizar_product_id(df["product_id"])
        df["forecast_company"] = pd.to_numeric(df["forecast_company"], errors="coerce").fillna(0)
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

        df = (
            df.groupby(["product_id", "date"], as_index=False)["forecast_company"]
            .sum()
            .sort_values(["product_id", "date"])
            .reset_index(drop=True)
        )
        return df

    return pd.DataFrame(columns=["date", "product_id", "forecast_company"])


def calcular_metricas(real, pred):
    real = pd.to_numeric(real, errors="coerce").fillna(0)
    pred = pd.to_numeric(pred, errors="coerce").fillna(0)
    suma_real = real.sum()
    mae = (pred - real).abs().mean() if len(real) else 0
    wmape = (pred - real).abs().sum() / suma_real if suma_real > 0 else 0
    bias = (pred - real).sum() / suma_real if suma_real > 0 else 0
    return wmape, bias, mae


def preparar_comparacion(df_real, df_forecast_empresa, df_forecast_auto, producto):
    producto_norm = normalizar_product_id(pd.Series([producto])).iloc[0]

    real = df_real.copy()
    prop = df_forecast_auto[df_forecast_auto.get("tipo_periodo", "Histórico") == "Histórico"].copy()
    emp = df_forecast_empresa.copy()

    for df in [real, prop, emp]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df["product_id"] = normalizar_product_id(df["product_id"])

    real = real[real["product_id"] == producto_norm]
    prop = prop[prop["product_id"] == producto_norm]
    emp = emp[emp["product_id"] == producto_norm]

    df = real[["date", "product_id", "demand_real"]].merge(
        emp[["date", "product_id", "forecast_company"]],
        on=["date", "product_id"],
        how="inner",
    )

    df = df.merge(
        prop[["date", "product_id", "demand_forecast", "method_used"]],
        on=["date", "product_id"],
        how="inner",
    )

    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(
        columns={
            "demand_real": "Venta real",
            "forecast_company": "Forecast comercial",
            "demand_forecast": "Forecast propuesto",
            "method_used": "Método propuesto",
        }
    )
    return df


archivo = st.sidebar.file_uploader("Sube el mismo Excel", type=["xlsx", "xls"])

if archivo is None:
    st.info("Sube tu Excel con las hojas Demanda y Forecast_Comercial.")
    st.stop()

try:
    xls = pd.ExcelFile(archivo)
    df_real = leer_demanda(xls)
    df_forecast_empresa = leer_forecast_comercial(xls)

    if df_forecast_empresa.empty:
        st.error("No se encontró la hoja Forecast_Comercial o sus columnas date, product_id y forecast_company.")
        st.stop()

    ultima_fecha = pd.to_datetime(df_real["date"].max()).to_period("M").to_timestamp()
    fecha_fin = st.sidebar.date_input(
        "Pronosticar hasta",
        value=pd.Timestamp("2026-12-01"),
        min_value=ultima_fecha.date(),
    )
    fecha_fin = pd.to_datetime(fecha_fin).to_period("M").to_timestamp()

    with st.spinner("Calculando forecast propuesto por SKU..."):
        df_forecast_auto, df_comparacion = generar_forecast_mejor_por_producto(
            df_real,
            fecha_fin_pronostico=fecha_fin,
        )

except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()

productos_real = set(normalizar_product_id(df_real["product_id"]))
productos_emp = set(normalizar_product_id(df_forecast_empresa["product_id"]))
productos_prop = set(normalizar_product_id(df_forecast_auto["product_id"]))
productos = sorted(productos_real & productos_emp & productos_prop)

if not productos:
    st.error("No hay SKUs que crucen entre demanda real, forecast comercial y forecast propuesto.")
    st.stop()

producto_sel = st.sidebar.selectbox("SKU a comparar", productos)

df_comp = preparar_comparacion(df_real, df_forecast_empresa, df_forecast_auto, producto_sel)

if df_comp.empty:
    st.warning("Este SKU no tiene coincidencias por mes entre las tres fuentes.")
    st.stop()

metodo = df_comp["Método propuesto"].iloc[0]
wmape_emp, bias_emp, mae_emp = calcular_metricas(df_comp["Venta real"], df_comp["Forecast comercial"])
wmape_prop, bias_prop, mae_prop = calcular_metricas(df_comp["Venta real"], df_comp["Forecast propuesto"])
mejora_wmape = wmape_emp - wmape_prop

c1, c2, c3, c4 = st.columns(4)
c1.metric("Método propuesto", metodo)
c2.metric("wMAPE comercial", f"{wmape_emp:.2%}")
c3.metric("wMAPE propuesto", f"{wmape_prop:.2%}", delta=f"{mejora_wmape:.2%}")
c4.metric("Meses comparados", f"{len(df_comp):,}")

st.divider()

st.subheader("📊 Comparación visual de los 3 forecasts")

plot_df = df_comp[["date", "Venta real", "Forecast comercial", "Forecast propuesto"]].melt(
    id_vars="date",
    var_name="Serie",
    value_name="Unidades",
)

fig = px.line(
    plot_df,
    x="date",
    y="Unidades",
    color="Serie",
    markers=True,
    title=f"Venta real vs Forecast comercial vs Forecast propuesto - {producto_sel}",
)
fig.update_layout(
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Tabla mensual")
tabla = df_comp.copy()
tabla["Mes"] = pd.to_datetime(tabla["date"]).dt.strftime("%b %Y").str.upper()
tabla = tabla[["Mes", "Venta real", "Forecast comercial", "Forecast propuesto", "Método propuesto"]]

st.dataframe(
    tabla.style.format({
        "Venta real": "{:,.0f}",
        "Forecast comercial": "{:,.0f}",
        "Forecast propuesto": "{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "📥 Descargar comparación CSV",
    data=df_comp.to_csv(index=False).encode("utf-8"),
    file_name=f"comparacion_forecast_{producto_sel[:30]}.csv",
    mime="text/csv",
)
