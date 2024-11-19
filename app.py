import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pulp as pl

# Parámetros fijos
centros_distribucion = {"CD Lampa": 1, "ABCPACK": 2, "Egakat": 3}
capacidad_camion = 28
coste_por_camion = 70000
penalizacion_retraso = 70000
capacidad_horaria = 2
horarios = list(range(24))

# Matriz de probabilidades de retraso (hora x día)
prob_retraso = [
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.5, 0.5, 0.5, 0.5, 0.5, 0.33, 1.0],
  [0.5, 0.5, 0.5, 0.5, 0.5, 0.33, 1.0],
  [0.5, 0.5, 0.5, 0.5, 0.5, 0.33, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  [0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0],
  [0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0],
  [0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
  ]

# Navegación entre versiones
st.title("Optimización de Despacho de Camiones")
opcion = st.radio("Selecciona la versión de la aplicación:", ["Semanal", "Diaria"])

# Versión diaria
if opcion == "Diaria":
    st.header("Optimización diaria")

    # Selección del día
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_seleccionado = st.selectbox("Selecciona el día:", dias)
    dia_idx = dias.index(dia_seleccionado)

    # Entrada de demanda diaria
    st.subheader("Demanda diaria por pallets")
    demanda_diaria_pallets = {}
    for centro in centros_distribucion.keys():
        demanda_diaria_pallets[centro] = st.number_input(
            f"Demanda para {centro}:", min_value=0, step=1
        )

    # Entrada de horarios ocupados
    st.subheader("Horarios ocupados")
    horarios_ocupados = st.text_input(
        "Introduce los horarios ocupados (separados por comas, por ejemplo: 10,11,12):",
        placeholder="Ejemplo: 10,11,12"
    )
    horarios_ocupados = list(map(int, horarios_ocupados.split(','))) if horarios_ocupados else []

    # Botón para ejecutar el modelo
    if st.button("Ejecutar modelo de optimización"):
        try:
            # Crear el modelo
            modelo = pl.LpProblem("Optimización_Diaria", pl.LpMinimize)
            x = pl.LpVariable.dicts("Camiones_pedidos", horarios, 0, None, pl.LpInteger)
            retraso = pl.LpVariable.dicts("Retraso", horarios, 0, None, pl.LpInteger)

            # Función objetivo
            modelo += pl.lpSum([penalizacion_retraso * retraso[t] for t in horarios]) + \
                      pl.lpSum([coste_por_camion * x[t] for t in horarios])

            # Restricciones
            # 1. Cumplir con la demanda diaria de pallets
            for centro, pallets in demanda_diaria_pallets.items():
                modelo += pl.lpSum([x[t] * capacidad_camion for t in horarios]) >= pallets, \
                          f"Satisfacer_demanda_{centro}"

            # 2. Restricción de horarios ocupados
            for t in horarios_ocupados:
                modelo += x[t] == 0, f"Horario_ocupado_{t}"

            # 3. Restricción de capacidad por hora
            for t in horarios:
                modelo += x[t] <= capacidad_horaria, f"Capacidad_max_por_hora_{t}"

            # 4. Restricción de probabilidades de retraso
            prob_retraso = [0.5 if t % 2 == 0 else 0.2 for t in horarios]  # Ejemplo simple de probabilidad
            for t in horarios:
                modelo += retraso[t] >= prob_retraso[t] * x[t], f"Retraso_por_probabilidad_{t}"

            # Resolver el modelo
            modelo.solve()

            # Generar el calendario para el día
            calendario = pd.DataFrame(index=[f"{h}:00" for h in horarios],
                                       columns=["Camiones"])
            for t in horarios:
                calendario.iloc[t, 0] = x[t].varValue if x[t].varValue > 0 else 0

            # Visualización del gráfico
            st.subheader("Calendario de camiones")
            plt.figure(figsize=(10, 6))
            sns.heatmap(calendario.astype(float), annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Cantidad de Camiones'})
            plt.title(f"Calendario de Camiones - {dia_seleccionado}")
            plt.ylabel("Horas")
            plt.xlabel("Camiones")
            st.pyplot(plt)

            # Generar reporte de retrasos
            st.subheader("Reporte de retrasos")
            retrasos_df = pd.DataFrame({"Hora": [f"{h}:00" for h in horarios],
                                        "Retrasos": [retraso[t].varValue for t in horarios]})
            retrasos_csv = retrasos_df.to_csv(index=False)
            st.download_button("Descargar reporte de retrasos", data=retrasos_csv, file_name=f"retrasos_{dia_seleccionado}.csv")

        except Exception as e:
            st.error(f"Error: {e}")

# Versión semanal (placeholder para coexistencia)
if opcion == "Semanal":
    st.header("Optimización semanal")
    st.write("La versión semanal sigue disponible en esta misma aplicación.")
    # Parámetros fijos
    centros_distribucion = {"CD Lampa": 1, "ABCPACK": 2, "Egakat": 3}
    capacidad_camion = 28
    coste_por_camion = 70000
    penalizacion_retraso = 70000
    capacidad_horaria = 2
    horarios = list(range(24))
    dias = list(range(7))
    
    # Título de la aplicación
    st.title("Optimización de Despacho de Camiones")
    
    # Entrada de datos
    st.header("1. Introduce los datos necesarios")
    
    # Input: Demanda diaria
    st.subheader("Demanda diaria por pallets")
    demanda_diaria_pallets = {}
    for centro in centros_distribucion.keys():
        demanda_diaria_pallets[centro] = st.text_input(
            f"Demanda semanal para {centro} (separada por comas, ejemplo: 100,200,300,400,500,0,0):",
            placeholder="Introduce 7 valores separados por comas"
        )
    
    # Input: Pallets pendientes
    st.subheader("Pallets pendientes")
    pallets_pendientes = {}
    for centro in centros_distribucion.keys():
        pallets_pendientes[centro] = st.number_input(
            f"Pallets pendientes para {centro}:",
            min_value=0,
            step=1
        )
    
    # Input: Horarios ocupados diarios
    st.subheader("Horarios ocupados diarios")
    horarios_ocupados_diarios = {}
    for dia in dias:
        horarios_ocupados_diarios[dia] = st.text_input(
            f"Horarios ocupados para el día {['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][dia]} (separados por comas):",
            placeholder="Ejemplo: 10,11,12"
        )
    
    # Botón para ejecutar el modelo
    if st.button("Ejecutar modelo de optimización"):
        try:
            # Procesar los inputs
            demanda_diaria_pallets = {k: list(map(int, v.split(','))) for k, v in demanda_diaria_pallets.items()}
            horarios_ocupados_diarios = {k: list(map(int, v.split(','))) if v else [] for k, v in horarios_ocupados_diarios.items()}
            
            # Crear el modelo
            modelo = pl.LpProblem("Optimización_Despacho_Camiones", pl.LpMinimize)
            x = pl.LpVariable.dicts("Camiones_pedidos", [(t, d) for t in horarios for d in dias], 0, None, pl.LpInteger)
            y = pl.LpVariable.dicts("Camiones_asignados", centros_distribucion, 0, None, pl.LpInteger)
            retraso = pl.LpVariable.dicts("Retraso", [(t, d) for t in horarios for d in dias], 0, None, pl.LpInteger)
    
            # Función objetivo
            modelo += pl.lpSum([penalizacion_retraso * retraso[(t, d)] for t in horarios for d in dias]) + \
                      pl.lpSum([coste_por_camion * x[(t, d)] for t in horarios for d in dias])
    
            # Restricciones
            # 1. Cumplir con la producción de pallets para cada centro diariamente (incluyendo pallets pendientes para el lunes)
            for j in centros_distribucion:
                for d in dias:
                    demanda = demanda_diaria_pallets[j][d]
                    if d == 0:  # Lunes
                        demanda += pallets_pendientes[j]
                    modelo += y[j] * capacidad_camion >= demanda, f"Satisfacer_demanda_centro_{j}_dia_{d}"
    
            # 2. Restricción de horarios ocupados por día
            for d in dias:
                for t in horarios_ocupados_diarios[d]:
                    modelo += x[(t, d)] == 0, f"Horario_ocupado_{t}_dia_{d}"
    
            # 3. Restricción de capacidad de 2 camiones por hora (con flexibilidad para lunes)
            for d in dias:
                for t in horarios:
                    if d == 0:  # Lunes
                        modelo += x[(t, d)] <= capacidad_horaria + 2, f"Capacidad_max_por_hora_flexible_{t}_dia_{d}"
                    else:
                        modelo += x[(t, d)] <= capacidad_horaria, f"Capacidad_max_por_hora_{t}_dia_{d}"
    
            # 4. Restricción de horas alternas
            for d in dias:
                for t in range(len(horarios) - 2):
                    modelo += x[(t, d)] + x[(t + 1, d)] <= capacidad_horaria, f"Restriccion_horas_alternas_{t}_dia_{d}"
    
            # 5. Satisfacer la demanda diaria de pallets entre todos los centros
            for d in dias:
                modelo += pl.lpSum([x[(t, d)] * capacidad_camion for t in horarios]) >= \
                          sum(demanda_diaria_pallets[j][d] for j in centros_distribucion), f"Satisfacer_demanda_total_dia_{d}"
    
            # 6. Vincular el número total de camiones pedidos con los camiones asignados a centros de distribución
            modelo += pl.lpSum([x[(t, d)] for t in horarios for d in dias]) == \
                      pl.lpSum([y[j] for j in centros_distribucion]), "Igualar_cantidad_pedidos_asignados"
    
            # 7. Restricción de probabilidad de retraso según el día y la hora
            for d in dias:
                for t in horarios:
                    modelo += retraso[(t, d)] >= prob_retraso[t][d] * x[(t, d)], f"Retraso_por_probabilidad_{t}_dia_{d}"
    
            # Resolver el modelo
            modelo.solve()
    
            # Generar el calendario
            calendario_camiones = pd.DataFrame(index=[f"{h}:00" for h in horarios],
                                               columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
            for d in dias:
                for t in horarios:
                    calendario_camiones.iloc[t, d] = x[(t, d)].varValue if x[(t, d)].varValue > 0 else 0
    
            # Visualización del gráfico
            st.header("2. Calendario de camiones")
            plt.figure(figsize=(10, 6))
            sns.heatmap(calendario_camiones.astype(float), annot=True, fmt=".0f", cmap="YlGnBu")
            plt.title("Calendario Semanal de Reserva de Camiones")
            plt.ylabel("Horas")
            plt.xlabel("Días de la Semana")
            st.pyplot(plt)
    
        except Exception as e:
            st.error(f"Error: {e}")
