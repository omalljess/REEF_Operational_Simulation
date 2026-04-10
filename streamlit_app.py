import streamlit as st
import polars as pl
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta, datetime # Import datetime
from PIL import Image

# --- Configuration --- #
base_data_path = r"/content/drive/MyDrive/REEF_Metocean_Data_R2/TT04-Example_B_2018_PT-varyingWL_cdeInput"

# Expected column names in the CSV files. Adjust these if your actual files use different names.
TIME_COL = "Time [s]"
X_EXCURSION_COL = "X Axis Excursion [m]"
Y_EXCURSION_COL = "Y Axis Excursion [m]"
Z_EXCURSION_COL = "Z Axis Excursion [m]"
X_VELOCITY_COL = "X Axis Velocity [m/s]"
Y_VELOCITY_COL = "Y Axis Velocity [m/s]"
Z_VELOCITY_COL = "Z Axis Velocity [m/s]"
SIM_WATER_SURFACE_Z_COL = "Sim WaterSurfaceZ [m]"
STATIC_WATER_SURFACE_Z_COL = "Static WaterSurfaceZ [m]"
MLAB_X_POSITION = "MLAB X Position [m]"
MLAB_Y_POSITION = "MLAB Y Position [m]"
MLAB_Z_POSITION = "MLAB Z Position [m]"
MLAC_X_POSITION = "MLAC X Position [m]"
MLAC_Y_POSITION = "MLAC Y Position [m]"
MLAC_Z_POSITION = "MLAC Z Position [m]"

# New columns for Link MB data
LINK_MB1_COL = "Link MB1 to BW Odin [kN]"
LINK_MB2_COL = "Link MB2 to BW Odin [kN]"
LINK_MB3_COL = "Link MB3 to BW Odin [kN]"
LINK_MB4_COL = "Link MB4 to BW Odin [kN]"
LINK_MB5_COL = "Link MB5 to BW Odin [kN]"

# All potentially desired columns
potential_cols = [
    TIME_COL,
    X_EXCURSION_COL,
    Y_EXCURSION_COL,
    Z_EXCURSION_COL,
    X_VELOCITY_COL,
    Y_VELOCITY_COL,
    Z_VELOCITY_COL,
    SIM_WATER_SURFACE_Z_COL,
    STATIC_WATER_SURFACE_Z_COL,
    LINK_MB1_COL,
    LINK_MB2_COL,
    LINK_MB3_COL,
    LINK_MB4_COL,
    LINK_MB5_COL,
    MLAB_X_POSITION,
    MLAB_Y_POSITION,
    MLAB_Z_POSITION,
    MLAC_X_POSITION,
    MLAC_Y_POSITION,
    MLAC_Z_POSITION
]

# Define a color palette for Link MB plots
LINK_MB_COLORS = ['blue', 'orange', 'green', 'red', 'purple']

# Define custom y-ticks and labels for wind direction
wind_dir_ticks_excel = [0, 45, 90, 135, 180, 225, 270, 315, 360]
wind_dir_labels_excel = ['N (0°)', 'NE (45°)', 'E (90°)', 'SE (135°)', 'S (180°)', 'SW (225°)', 'W (270°)', 'NW (315°)', 'N (360°)']

excel_data_configs = {
    "High-Risk-varyingWL-Example_B": {
        "file": r"/content/drive/MyDrive/REEF_Metocean_Data_R2/ExampleB/Non Prevailing Wind Examples.xlsx",
        "sheet_name": 0, # Assuming the first sheet
        "time_col": "Time [UTC]",
        "knots_col": "Knots",
        "udir_col": "Udir [° TN]",
        "hs_col": "Sea Hs [m]",
        "tp_col": "Sea Tp [s]"
    },
    "Med-Risk-varyingWL-HR": {
        "file": r"/content/drive/MyDrive/REEF_Metocean_Data_R2/WeatherScenarios/Weather Scenarios 2017 BMC Test Drive.xlsx",
        "sheet_name": "Med Risk",
        "time_col": "Time [UTC]",
        "knots_col": "Holland Rock Knots",
        "udir_col": "Udir [° TN]",
        "hs_col": "Sea Hs [m]",
        "tp_col": "Sea Tp [s]"
    },
    "High-Risk-varyingWL-HR": {
        "file": r"/content/drive/MyDrive/REEF_Metocean_Data_R2/WeatherScenarios/Weather Scenarios 2017 BMC Test Drive.xlsx",
        "sheet_name": "High Risk",
        "time_col": "Time [UTC]",
        "knots_col": "Holland Rock Knots",
        "udir_col": "Udir [° TN]",
        "hs_col": "Sea Hs [m]",
        "tp_col": "Sea Tp [s]"
    },
    "High-Risk-varyingWL-NonHR": {
        "file": r"/content/drive/MyDrive/REEF_Metocean_Data_R2/WeatherScenarios/Weather Scenarios 2017 BMC Test Drive.xlsx",
        "sheet_name": "High Risk",
        "time_col": "Time [UTC]",
        "knots_col": "Knots", # Specific column for this folder
        "udir_col": "Udir [° TN]",
        "hs_col": "Sea Hs [m]",
        "tp_col": "Sea Tp [s]"
    }
}

# --- Data Loading and Preprocessing --- #
@st.cache_data
def load_data():
    dataset_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    all_dataset_folder_dfs = []

    for folder_path in dataset_folders:
        folder_name = os.path.basename(folder_path)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            continue

        first_file_path = csv_files[0]
        try:
            temp_df = pl.read_csv(first_file_path, infer_schema_length=0, n_rows=0)
            available_cols = temp_df.columns
            current_selected_cols = [col for col in potential_cols if col in available_cols]
        except Exception:
            continue

        if not current_selected_cols:
            continue

        current_folder_file_dfs = []
        for file_path in csv_files:
            try:
                df_file = pl.read_csv(file_path, columns=current_selected_cols, infer_schema_length=1000)
                date_identifier = os.path.basename(file_path).replace(".csv", "")
                df_file = df_file.with_columns(pl.lit(date_identifier).alias("file_date"))
                current_folder_file_dfs.append(df_file)
            except Exception:
                continue

        if current_folder_file_dfs:
            try:
                folder_compiled_df = pl.concat(current_folder_file_dfs, how="vertical")

                # Apply excursion remapping and sign adjustments within the loading function
                if X_EXCURSION_COL in folder_compiled_df.columns and \
                   Y_EXCURSION_COL in folder_compiled_df.columns and \
                   Z_EXCURSION_COL in folder_compiled_df.columns:

                    folder_compiled_df = folder_compiled_df.with_columns(
                        pl.col(X_EXCURSION_COL).alias("_temp_original_X_excursion"),
                        pl.col(Y_EXCURSION_COL).alias("_temp_original_Y_excursion"),
                        pl.col(Z_EXCURSION_COL).alias("_temp_original_Z_excursion")
                    )

                    mlaB_flange_x0 = 22.285
                    mlaB_flange_y0 = 0
                    mlaB_flange_z0 = -1.125

                    mlaC_flange_x0 = 22.285
                    mlaC_flange_y0 = 0
                    mlaC_flange_z0 = +1.125

                    folder_compiled_df = folder_compiled_df.with_columns(
                        pl.col("_temp_original_Z_excursion").alias(Y_EXCURSION_COL),
                        (pl.col("_temp_original_Y_excursion") * -1).alias(X_EXCURSION_COL),
                        (pl.col("_temp_original_X_excursion") * -1).alias(Z_EXCURSION_COL)
                    )

                    folder_compiled_df = folder_compiled_df.drop(["_temp_original_X_excursion", "_temp_original_Y_excursion", "_temp_original_Z_excursion"])

                    if SIM_WATER_SURFACE_Z_COL in folder_compiled_df.columns:
                        folder_compiled_df = folder_compiled_df.with_columns([
                            # MLAB
                            (pl.lit(mlaB_flange_x0) + pl.col(X_EXCURSION_COL)).alias(MLAB_X_POSITION),
                            (pl.lit(mlaB_flange_y0) + pl.col(Y_EXCURSION_COL) + pl.col(SIM_WATER_SURFACE_Z_COL)).alias(MLAB_Y_POSITION),
                            (pl.lit(mlaB_flange_z0) + pl.col(Z_EXCURSION_COL)).alias(MLAB_Z_POSITION),

                            # MLAC
                            (pl.lit(mlaC_flange_x0) + pl.col(X_EXCURSION_COL)).alias(MLAC_X_POSITION),
                            (pl.lit(mlaC_flange_y0) + pl.col(Y_EXCURSION_COL) + pl.col(SIM_WATER_SURFACE_Z_COL)).alias(MLAC_Y_POSITION),
                            (pl.lit(mlaC_flange_z0) + pl.col(Z_EXCURSION_COL)).alias(MLAC_Z_POSITION),
                        ])

                all_dataset_folder_dfs.append((folder_name, folder_compiled_df))
            except Exception:
                pass
    return dataset_folders, all_dataset_folder_dfs

@st.cache_data
def process_folder_data(df_folder_compiled, folder_name, original_dataset_folders):
    # Ensure data is sorted by file_date and then by Time [s] within each file
    df_sorted = df_folder_compiled.sort(['file_date', TIME_COL])

    # Get unique file dates and their maximum times
    file_end_times = df_sorted.group_by('file_date').agg(
        pl.col(TIME_COL).max().alias('max_time')
    ).sort('file_date')

    # Calculate cumulative time offsets
    file_end_times = file_end_times.with_columns(
        (pl.col('max_time').shift(1).fill_null(0.0)).alias('shifted_max_time')
    )
    shifted_max_time_np = file_end_times['shifted_max_time'].to_numpy()
    cumulative_offset_np = np.cumsum(shifted_max_time_np)

    file_end_times = file_end_times.with_columns(
        pl.Series("time_offset", cumulative_offset_np).alias('time_offset')
    )

    df_continuous = df_sorted.join(file_end_times.select(['file_date', 'time_offset']), on='file_date', how='left')

    df_continuous = df_continuous.with_columns(
        (pl.col(TIME_COL) + pl.col('time_offset')).alias('continuous_time_s')
    )
    df_continuous = df_continuous.with_columns(
        (pl.col('continuous_time_s') / (3600 * 24)).alias('continuous_time_d')
    )

    first_file_date_str_in_folder = df_sorted['file_date'].head(1).item()
    folder_start_datetime = pd.to_datetime(first_file_date_str_in_folder, format='%Y%m%d_%H%M%S')

    return df_continuous, folder_start_datetime

# --- Streamlit App --- #
st.set_page_config(layout="wide")
st.title("REEF Tabletop Exercise")

# Load all data
with st.spinner("Loading data, this may take a moment..."):
    original_dataset_folders, all_dataset_folder_data = load_data()

if not all_dataset_folder_data:
    st.error("No data found or loaded. Please check base_data_path and CSV files.")
    st.stop()

folder_names = [name for name, _ in all_dataset_folder_data]
selected_folder_name = st.sidebar.selectbox(
    "Select a Dataset Folder:",
    folder_names
)

selected_df_tuple = next((item for item in all_dataset_folder_data if item[0] == selected_folder_name), None)

if selected_df_tuple is None:
    st.error("Selected folder data not found.")
    st.stop()

df_selected_compiled = selected_df_tuple[1]

df_continuous, folder_start_datetime = process_folder_data(df_selected_compiled, selected_folder_name, original_dataset_folders)

min_continuous_time_s = df_continuous['continuous_time_s'].min()
max_continuous_time_s = df_continuous['continuous_time_s'].max()

min_continuous_time_d = df_continuous['continuous_time_d'].min()
max_continuous_time_d = df_continuous['continuous_time_d'].max()

# total_days is the number of full days covered by the data.
# If data runs from 0.0 to 7.0 days, this is 7 full days (Day 1 to Day 7).
# The slider should therefore range from 1 to total_days.
total_days_for_slider = int(np.ceil(max_continuous_time_d - min_continuous_time_d))

st.sidebar.header("Time Selection")
# Change slider to be 1-indexed for user-friendliness
selected_day = st.sidebar.slider("Select Day", 1, total_days_for_slider, 1)
selected_hour = st.sidebar.slider("Select Hour (0-23)", 0, 23, 0)
selected_minute = st.sidebar.slider("Select Minute (0-59)", 0, 59, 0)
selected_second = st.sidebar.slider("Select Second (0-59)", 0, 59, 0)

# Calculate selected time in continuous days (adjust for 1-indexed slider)
selected_time_d = min_continuous_time_d + (selected_day - 1) + (selected_hour / 24.0) + (selected_minute / (24.0 * 60.0)) + (selected_second / (24.0 * 60 * 60))

# Add zoom toggle button
#zoom_to_selected_time = st.sidebar.checkbox("Zoom to +/- 12 hours around selected time")
zoom_to_selected_time = False

# Add hide future data toggle button
# hide_future_data = st.sidebar.checkbox("Only Past Wind")
hide_future_data = False

# Add toggle for conditionally smoothed wind data
show_conditionally_smoothed_wind = st.sidebar.checkbox("Show Wind Forecast")

# Determine x-axis limits based on zoom toggle and hide future data toggle
if zoom_to_selected_time:
    # 12 hours in days = 0.5 days
    x_lim_min = selected_time_d - 0.5
    x_lim_max = selected_time_d + 0.5
    # Ensure limits don't go beyond available data range
    x_lim_min = max(x_lim_min, min_continuous_time_d)
    x_lim_max = min(x_lim_max, max_continuous_time_d)
else:
    x_lim_min = min_continuous_time_d
    x_lim_max = max_continuous_time_d

# Apply hide future data if enabled
if hide_future_data:
      x_lim_max = min(x_lim_max, selected_time_d) # Cap the max limit at selected time

# Generate tick locations and labels for x-axis (4-hour intervals for wind plots)
four_hour_in_seconds = 4 * 3600
start_tick_s = np.floor(min_continuous_time_s / four_hour_in_seconds) * four_hour_in_seconds
tick_locs_s = np.arange(start_tick_s, max_continuous_time_s + four_hour_in_seconds, four_hour_in_seconds)
tick_locs_d = tick_locs_s / (3600 * 24)

tick_labels = []
for t_s in tick_locs_s:
    day = int(t_s / (3600 * 24)) + 1 # Add 1 to make D0 start as D1
    hour = int((t_s % (3600 * 24)) / 3600)
    tick_labels.append(f'D{day} H{hour:02d}')

# Filter ticks and labels to be within the determined x_lim_min and x_lim_max for wind plots
filtered_tick_locs_d = [t for t in tick_locs_d if x_lim_min <= t <= x_lim_max]
filtered_tick_labels = [tick_labels[i] for i, t in enumerate(tick_locs_d) if x_lim_min <= t <= x_lim_max]

# Define 4-hour window for excursion and link tension plots
x_lim_min_4h = max(selected_time_d - (4 / 24.0), min_continuous_time_d)
x_lim_max_4h = selected_time_d

# Generate tick locations in seconds, every 1 hour, for the full range
one_hour_in_seconds = 1 * 3600
start_tick_s_1h = np.floor(min_continuous_time_s / one_hour_in_seconds) * one_hour_in_seconds
tick_locs_s_1h_full = np.arange(start_tick_s_1h, max_continuous_time_s + one_hour_in_seconds, one_hour_in_seconds)
tick_locs_d_1h_full = tick_locs_s_1h_full / (3600 * 24)

# Generate tick labels in 'Day X Hour Y' format for the full range
tick_labels_1h_full = []
for t_s in tick_locs_s_1h_full:
    day = int(t_s / (3600 * 24)) + 1
    hour = int((t_s % (3600 * 24)) / 3600)
    tick_labels_1h_full.append(f'D{day} H{hour:02d}')

# Filter 1-hour ticks and labels to be within the 4-hour window
filtered_tick_locs_d_1h = [t for t in tick_locs_d_1h_full if x_lim_min_4h <= t <= x_lim_max_4h]
filtered_tick_labels_1h = [tick_labels_1h_full[i] for i, t in enumerate(tick_locs_d_1h_full) if x_lim_min_4h <= t <= x_lim_max_4h]


# --- Get exact MLAB + MLAC positions at selected time ---
mlab_row = (
    df_continuous
    .with_columns((pl.col('continuous_time_d') - selected_time_d).abs().alias('dt'))
    .sort('dt')
    .select([
        'continuous_time_d',
        MLAB_X_POSITION, MLAB_Y_POSITION, MLAB_Z_POSITION,
        MLAC_X_POSITION, MLAC_Y_POSITION, MLAC_Z_POSITION
    ])
    .row(0)
)

(actual_time_d,
 mlab_x, mlab_y, mlab_z,
 mlac_x, mlac_y, mlac_z) = mlab_row


# --- Plotting --- #
plot_cols_main = [
    (X_EXCURSION_COL, X_EXCURSION_COL, 'blue', 'Position [m]'),
    (Y_EXCURSION_COL, Y_EXCURSION_COL, 'green', 'Position [m]'),
    (Z_EXCURSION_COL, Z_EXCURSION_COL, 'red', 'Position [m]'),
    (SIM_WATER_SURFACE_Z_COL, SIM_WATER_SURFACE_Z_COL, 'cyan', 'Position [m]')
]

mb_cols_present = all(col in df_continuous.columns for col in [LINK_MB1_COL, LINK_MB2_COL, LINK_MB3_COL, LINK_MB4_COL, LINK_MB5_COL])

# --- Pre-calculate stats for Excursion/Water Surface plots ---
excursion_stats = {}
for original_col, plot_col, color, ylabel in plot_cols_main:
    val_at_time = (
        df_continuous
        .with_columns((pl.col('continuous_time_d') - selected_time_d).abs().alias('dt'))
        .sort('dt')
        .select(plot_col)
        .row(0)[0]
    )

    max_val_4h = (
        df_continuous
        .filter(
            (pl.col('continuous_time_d') >= x_lim_min_4h) &
            (pl.col('continuous_time_d') <= x_lim_max_4h)
        )
        .with_columns(pl.col(plot_col).abs().alias('abs_val'))
        .sort('abs_val', descending=True)
        .select(plot_col)
        .row(0)[0]
    )

    excursion_stats[original_col] = {
        'val_at_time': val_at_time,
        'max_val_4h': max_val_4h,
        'ylabel': ylabel
    }

# --- Pre-calculate stats for Link MB Force plots ---
link_mb_stats = {}
if "varyingWL" in selected_folder_name and mb_cols_present:
    for k, link_col in enumerate([LINK_MB1_COL, LINK_MB2_COL, LINK_MB3_COL, LINK_MB4_COL, LINK_MB5_COL]):
        smoothed_col = link_col # Assuming no smoothing for stats display
        val_at_time = (
            df_continuous
            .with_columns((pl.col('continuous_time_d') - selected_time_d).abs().alias('dt'))
            .sort('dt')
            .select(smoothed_col)
            .row(0)[0]
        )
        max_val_4h = df_continuous.filter(
            (pl.col('continuous_time_d') >= x_lim_min_4h) &
            (pl.col('continuous_time_d') <= x_lim_max_4h)
        ).select(smoothed_col).max().item()
        link_mb_stats[link_col] = {
            'val_at_time': val_at_time,
            'max_val_4h': max_val_4h
        }

# External Data Plots (Wind)
st.header("Current Wind Conditions")
if selected_folder_name in excel_data_configs:
    config = excel_data_configs[selected_folder_name]
    current_excel_file_path = config["file"]
    current_excel_sheet_name = config["sheet_name"]
    current_time_col_excel = config["time_col"]
    current_knots_col_excel = config["knots_col"]
    current_udir_col_excel = config["udir_col"]
    current_hs_col_excel = config.get("hs_col")
    current_tp_col_excel = config.get("tp_col")

    current_df_excel = None
    current_min_timestamp_excel = None

    try:
        current_df_excel_pd = pd.read_excel(current_excel_file_path, sheet_name=current_excel_sheet_name)
        current_df_excel = pl.from_pandas(current_df_excel_pd)
    except Exception as e:
        st.error(f"Error loading Excel file for {selected_folder_name}: {e}")

    if current_df_excel is not None:
        if current_time_col_excel not in current_df_excel.columns:
            st.error(f"Error: '{current_time_col_excel}' not found in Excel data for {selected_folder_name}.")
            current_df_excel = None
        else:
            current_min_timestamp_excel = current_df_excel[current_time_col_excel].min()
            current_df_excel = current_df_excel.with_columns(
                ((pl.col(current_time_col_excel) - current_min_timestamp_excel).dt.total_seconds()).alias('continuous_time_s_excel')
            )
            current_df_excel = current_df_excel.with_columns(
                (pl.col('continuous_time_s_excel') / (3600 * 24)).alias('continuous_time_d_excel')
            )

            missing_cols = [
                col for col in [
                    current_knots_col_excel,
                    current_udir_col_excel,
                    current_hs_col_excel,
                    current_tp_col_excel
                ]
                if col is not None and col not in current_df_excel.columns
            ]

            if missing_cols:
                st.error(f"Missing columns in Excel data for {selected_folder_name}: {missing_cols}")
                current_df_excel = None

    if current_df_excel is not None:
        time_shift_days = (folder_start_datetime - current_min_timestamp_excel).total_seconds() / (3600 * 24)

        df_excel_with_shifted_time = current_df_excel.with_columns(
            (pl.col('continuous_time_d_excel') + time_shift_days).alias('absolute_continuous_time_d_excel')
        )

        df_excel_filtered_aligned = df_excel_with_shifted_time.filter(
            (pl.col('absolute_continuous_time_d_excel') >= min_continuous_time_d) &
            (pl.col('absolute_continuous_time_d_excel') <= max_continuous_time_d)
        )

        # ✅ Apply "Only Past Wind"
        if hide_future_data:
            df_excel_filtered_aligned = df_excel_filtered_aligned.filter(
                pl.col('absolute_continuous_time_d_excel') <= selected_time_d
            )

        if df_excel_filtered_aligned.is_empty():
            st.warning(f"No Excel data available for plotting for {selected_folder_name} within the time range.")
        else:
            # --- Plotting for raw wind data ---
            fig_excel, ax1_excel = plt.subplots(figsize=(18, 6))

            ax1_excel.plot(df_excel_filtered_aligned['absolute_continuous_time_d_excel'].to_numpy(), df_excel_filtered_aligned[current_knots_col_excel].to_numpy(),
                     label=current_knots_col_excel, color='blue', alpha=0.8)
            ax1_excel.set_xlabel('Time [d]', fontsize=10)
            ax1_excel.set_ylabel(current_knots_col_excel, color='blue', fontsize=10)
            ax1_excel.tick_params(axis='y', labelcolor='blue')
            ax1_excel.grid(True)
            ax1_excel.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1_excel.set_xticks(filtered_tick_locs_d)
            ax1_excel.set_xticklabels(filtered_tick_labels, rotation=45, ha='right')
            ax1_excel.set_xlim(x_lim_min, x_lim_max)
            ax1_excel.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')

            ax2_excel = ax1_excel.twinx()
            ax2_excel.scatter(df_excel_filtered_aligned['absolute_continuous_time_d_excel'].to_numpy(), df_excel_filtered_aligned[current_udir_col_excel].to_numpy(),
                     label=current_udir_col_excel, color='red', alpha=0.8, s=10)
            ax2_excel.set_ylabel('Wind Direction', color='red', fontsize=10)

            ax2_excel.set_yticks(wind_dir_ticks_excel)
            ax2_excel.set_yticklabels(wind_dir_labels_excel)
            ax2_excel.set_ylim(0, 360)

            ax2_excel.tick_params(axis='y', labelcolor='red')
            ax2_excel.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left')
            ax2_excel.set_xlim(x_lim_min, x_lim_max)
            ax2_excel.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')

            # st.pyplot(fig_excel)
            plt.close(fig_excel)

            val_knots_at_time = (
                df_excel_filtered_aligned
                .with_columns((pl.col('absolute_continuous_time_d_excel') - selected_time_d).abs().alias('dt'))
                .sort('dt')
                .select(current_knots_col_excel)
                .row(0)[0]
            )
            st.write(f"**Current Wind Speed:** {val_knots_at_time:.4f} Knots")

            val_udir_at_time = (
                df_excel_filtered_aligned
                .with_columns((pl.col('absolute_continuous_time_d_excel') - selected_time_d).abs().alias('dt'))
                .sort('dt')
                .select(current_udir_col_excel)
                .row(0)[0]
            )
            st.write(f"**Current Wind Direction:** {val_udir_at_time:.4f} °TN")

            # --- Sea state at selected time ---
            val_hs_at_time = None
            val_tp_at_time = None

            if current_hs_col_excel:
                val_hs_at_time = (
                    df_excel_filtered_aligned
                    .with_columns((pl.col('absolute_continuous_time_d_excel') - selected_time_d).abs().alias('dt'))
                    .sort('dt')
                    .select(current_hs_col_excel)
                    .row(0)[0]
                )

            if current_tp_col_excel:
                val_tp_at_time = (
                    df_excel_filtered_aligned
                    .with_columns((pl.col('absolute_continuous_time_d_excel') - selected_time_d).abs().alias('dt'))
                    .sort('dt')
                    .select(current_tp_col_excel)
                    .row(0)[0]
                )

            if val_hs_at_time is not None:
                st.write(f"**Current Wave Height:** {val_hs_at_time:.4f} m")

            if val_tp_at_time is not None:
                st.write(f"**Current Wave Period:** {val_tp_at_time:.4f} s")

            # --- Plotting for conditionally smoothed wind data ---
            if show_conditionally_smoothed_wind:
                st.subheader("Wind Forecast")

                # Apply Conditional Smoothing
                df_plot_smoothed = df_excel_filtered_aligned.with_columns([
                    pl.when(pl.col('absolute_continuous_time_d_excel') <= selected_time_d)
                    .then(pl.col(current_knots_col_excel))
                    .when((pl.col('absolute_continuous_time_d_excel') > selected_time_d) & (pl.col('absolute_continuous_time_d_excel') <= selected_time_d + 1)) # Next 24 hours
                    .then(pl.col(current_knots_col_excel).rolling_mean(window_size=2, center=True, min_periods=1))
                    .otherwise(pl.col(current_knots_col_excel).rolling_mean(window_size=10, center=True, min_periods=1))
                    .alias(f'{current_knots_col_excel}_smoothed'),

                    pl.when(pl.col('absolute_continuous_time_d_excel') <= selected_time_d)
                    .then(pl.col(current_udir_col_excel))
                    .when((pl.col('absolute_continuous_time_d_excel') > selected_time_d) & (pl.col('absolute_continuous_time_d_excel') <= selected_time_d + 1))
                    .then(pl.col(current_udir_col_excel).rolling_mean(window_size=2, center=True, min_periods=1))
                    .otherwise(pl.col(current_udir_col_excel).rolling_mean(window_size=10, center=True, min_periods=1))
                    .alias(f'{current_udir_col_excel}_smoothed')
                ])

                fig_smoothed_wind, ax1_smoothed_wind = plt.subplots(figsize=(18, 6))

                # Plot Knots on the first y-axis
                ax1_smoothed_wind.plot(df_plot_smoothed['absolute_continuous_time_d_excel'].to_numpy(), df_plot_smoothed[f'{current_knots_col_excel}_smoothed'].to_numpy(),
                                 label=f'{current_knots_col_excel}', color='blue', alpha=0.8)
                ax1_smoothed_wind.set_xlabel('Time [d]', fontsize=10)
                ax1_smoothed_wind.set_ylabel(current_knots_col_excel, color='blue', fontsize=10)
                ax1_smoothed_wind.tick_params(axis='y', labelcolor='blue')
                ax1_smoothed_wind.grid(True)
                ax1_smoothed_wind.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1_smoothed_wind.set_xticks(filtered_tick_locs_d)
                ax1_smoothed_wind.set_xticklabels(filtered_tick_labels, rotation=45, ha='right')
                ax1_smoothed_wind.set_xlim(x_lim_min, x_lim_max)
                ax1_smoothed_wind.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')

                # Create a second y-axis for Udir
                ax2_smoothed_wind = ax1_smoothed_wind.twinx()
                ax2_smoothed_wind.scatter(df_plot_smoothed['absolute_continuous_time_d_excel'].to_numpy(), df_plot_smoothed[f'{current_udir_col_excel}_smoothed'].to_numpy(),
                                 label=f'{current_udir_col_excel}', color='red', alpha=0.8, s=10)
                ax2_smoothed_wind.set_ylabel('Wind Direction', color='red', fontsize=10)
                ax2_smoothed_wind.set_yticks(wind_dir_ticks_excel)
                ax2_smoothed_wind.set_yticklabels(wind_dir_labels_excel)
                ax2_smoothed_wind.set_ylim(0, 360)
                ax2_smoothed_wind.tick_params(axis='y', labelcolor='red')
                ax2_smoothed_wind.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left')
                ax2_smoothed_wind.set_xlim(x_lim_min, x_lim_max)
                ax2_smoothed_wind.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')

                st.pyplot(fig_smoothed_wind)
                plt.close(fig_smoothed_wind)

                val_knots_at_time_smoothed = df_plot_smoothed.filter(
                    (pl.col('absolute_continuous_time_d_excel') >= selected_time_d - (0.5 / 24.0 / 60.0)) &
                    (pl.col('absolute_continuous_time_d_excel') <= selected_time_d + (0.5 / 24.0 / 60.0))
                ).select(f'{current_knots_col_excel}_smoothed').mean().item()
                #st.write(f"**{current_knots_col_excel} (Smoothed) at selected time:** {val_knots_at_time_smoothed:.4f}")

                val_udir_at_time_smoothed = df_plot_smoothed.filter(
                    (pl.col('absolute_continuous_time_d_excel') >= selected_time_d - (0.5 / 24.0 / 60.0)) &
                    (pl.col('absolute_continuous_time_d_excel') <= selected_time_d + (0.5 / 24.0 / 60.0))
                ).select(f'{current_udir_col_excel}_smoothed').mean().item()
                #st.write(f"**{current_udir_col_excel} (Smoothed) at selected time:** {val_udir_at_time_smoothed:.4f} °TN")

else:
    st.info(f"No Excel wind data configured for {selected_folder_name}")

# New Section: Link MB Force Values
if link_mb_stats: # Check if stats were computed (means varyingWL and mb_cols_present)
    st.header("Link MB Force Values")

    # Load images
    image2 = Image.open('Berth_op_controlloing_wind_speeds.png')
    image3 = Image.open('Mooring_layout.png')
    # Set desired width for both images (pixels)
    img_width = 400

    col1, col2 = st.columns(2)
    with col1:
      st.image(image2, width=img_width)
    with col2:
      st.image(image3, width=img_width)

    st.subheader("Values at Selected Time and Max in Last 4 Hours")
    for original_col, stats in link_mb_stats.items():
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{original_col} at selected time:** {stats['val_at_time']:.4f} kN")
        with col2:
            st.write(f"**Max {original_col} in last 4 hours:** {stats['max_val_4h']:.4f} kN")

# Existing Section: Link MB Force Plots
if "varyingWL" in selected_folder_name and mb_cols_present:
    st.header("Link MB Force Plots")
    plot_cols_mb_combined = []
    plot_cols_mb_individual = []
    for k, link_col in enumerate([LINK_MB1_COL, LINK_MB2_COL, LINK_MB3_COL, LINK_MB4_COL, LINK_MB5_COL]):
        smoothed_col = link_col
        plot_cols_mb_combined.append((link_col, smoothed_col, LINK_MB_COLORS[k]))
        plot_cols_mb_individual.append((link_col, smoothed_col, LINK_MB_COLORS[k]))

    # Combined Link MB plot
    fig, ax = plt.subplots(figsize=(18, 6))
    for original_col, plot_col, color in plot_cols_mb_combined:
        ax.plot(df_continuous['continuous_time_d'].to_numpy(), df_continuous[plot_col].to_numpy(),
                 label=f'{original_col}', color=color, alpha=0.8)

    ax.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')
    ax.axhline(y=1104, color='gray', linestyle='--', label='3-Line Max Tension (1104 kN)')
    ax.axhline(y=1472, color='darkgray', linestyle=':', label='4-Line Max Tension (1472 kN)')
    ax.set_title('All Link MB Forces vs. Time', fontsize=12)
    ax.set_xlabel('Time [d]', fontsize=10)
    ax.set_ylabel('Force [kN]', fontsize=10)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(filtered_tick_locs_d_1h)
    ax.set_xticklabels(filtered_tick_labels_1h, rotation=45, ha='right')
    ax.set_xlim(x_lim_min_4h, x_lim_max_4h)
    st.pyplot(fig)
    plt.close(fig)

    # Individual Link MB plots
    for original_col, plot_col, color in plot_cols_mb_individual:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(df_continuous['continuous_time_d'].to_numpy(), df_continuous[plot_col].to_numpy(),
                 label=f'{original_col}', color=color, alpha=0.8)
        ax.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')
        ax.axhline(y=1104, color='gray', linestyle='--', label='3-Line Max Tension (1104 kN)')
        ax.axhline(y=1472, color='darkgray', linestyle=':', label='4-Line Max Tension (1472 kN)')
        ax.set_title(f'{original_col} vs. Time', fontsize=12)
        ax.set_xlabel('Time [d]', fontsize=10)
        ax.set_ylabel('Force [kN]', fontsize=10)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks(filtered_tick_locs_d_1h)
        ax.set_xticklabels(filtered_tick_labels_1h, rotation=45, ha='right')
        ax.set_xlim(x_lim_min_4h, x_lim_max_4h)
        st.pyplot(fig)
        plt.close(fig)

# New Section: Excursion Values
if excursion_stats:
    st.header("Excursion Values")

    # Load images
    image1 = Image.open('ship_coordinates.png')
    # Set desired width for both images (pixels)
    img_width = 300
    st.image(image1, caption='Defined Coordinate System', width=img_width)

    st.subheader("Values at Selected Time and Max in Last 4 Hours")
    for original_col, stats in excursion_stats.items():
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{original_col} at selected time:** {stats['val_at_time']:.4f} {stats['ylabel'].split(' ')[-1]}")
        with col2:
            st.write(f"**Max {original_col} in last 4 hours:** {stats['max_val_4h']:.4f} {stats['ylabel'].split(' ')[-1]}")

# Existing Section: Excursion Plots
st.header("Excursion Plots") # Renamed from "Main Data Plots"
for original_col, plot_col, color, ylabel in plot_cols_main:
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(df_continuous['continuous_time_d'].to_numpy(), df_continuous[plot_col].to_numpy(),
             label=f'{original_col}', color=color, alpha=0.8)
    ax.axvline(x=selected_time_d, color='gray', linestyle='--', label=f'Selected Time: Day {selected_day} Hour {selected_hour:02d}')
    ax.set_title(f'{original_col} vs. Time', fontsize=12)
    ax.set_xlabel('Time [d]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(filtered_tick_locs_d_1h)
    ax.set_xticklabels(filtered_tick_labels_1h, rotation=45, ha='right')
    ax.set_xlim(x_lim_min_4h, x_lim_max_4h)
    st.pyplot(fig)
    plt.close(fig)

st.header("HMI DeltaV Screen")

st.markdown(
    "<h3 style='text-align: center;'>BERTH 1 91-MLA-9901B MARINE LOADING ARM POSITIONING SYSTEM 1 PROPANE</h3>",
    unsafe_allow_html=True
)

inboard_length = 17500/1000    #length of the inboard arm in m
riser_length = 9000/1000       #length of the MLA riser in m
outboard_length = 19250/1000   #length of the outboard arm in m

x_tsa_extension = 1040/1000
y_tsa_extension = 2185/1000
z_tsa_extension_B= 0.6192  # +0.6192 for MLA B, -0.6192 for MLA C
z_tsa_extension_C = -0.6192

x_CD_ext = .254

ingress_marine = 15.75
z_flange_dist = 2.25

#MLA & gangway origins
x_origin_B = + -3.5
z_origin_B = -2.500

x_origin_C = -3.5
z_origin_C = +2.500

offset_alarm = 7  #offset applied to acheive alarm function
offset_MSD2 = 2     #offset applied to acheive MSD2 function
offset_MSD1 = 1.5   #offset applied to acheive MSD1 function

#IF OFFSET VARRIES BY CIRCLE/LINE, INPUT DESIRED OFFSET FOR EACH BELOW
#pink offsets
offset_alarm_pink = -offset_alarm
offset_MSD2_pink = -offset_MSD2
offset_MSD1_pink = -offset_MSD1

#red offsets
offset_alarm_red = offset_alarm
offset_MSD2_red = offset_MSD2
offset_MSD1_red = offset_MSD1

#blue offsets
offset_alarm_blue = offset_alarm
offset_MSD2_blue = offset_MSD2
offset_MSD1_blue = offset_MSD1

#black offsets
offset_alarm_black = offset_alarm
offset_MSD2_black = offset_MSD2
offset_MSD1_black = offset_MSD1

#grey offsets
offset_alarm_ingress = offset_alarm
offset_MSD2_ingress = offset_MSD2
offset_MSD1_ingress = offset_MSD1

#equtions for circles
# Array t from 0 to 2*pi
t = np.linspace(0, 2*np.pi, 10000)

# Core function of all circles is (x+A^2) + (y+B^2) = (D +C) edit ithis

# Constants from pink circle
A_pink = -1.358533
B_pink = 24.26789
C_pink = 19.25250575

# Base functions
x_max_pink = A_pink + C_pink * np.cos(t)
y_max_pink = B_pink + C_pink * np.sin(t)

# MSD1:
x_msd1_pink = A_pink + (C_pink-offset_MSD1_pink) * np.cos(t)
y_msd1_pink = B_pink + (C_pink-offset_MSD1_pink) * np.sin(t)

# MSD2:
x_msd2_pink = A_pink + (C_pink-offset_MSD2_pink) * np.cos(t)
y_msd2_pink = B_pink + (C_pink-offset_MSD2_pink) * np.sin(t)

# ALARM:
x_alarm_pink = A_pink + (C_pink-offset_alarm_pink) * np.cos(t)
y_alarm_pink = B_pink + (C_pink-offset_alarm_pink) * np.sin(t)

# Constants for blue circle
A_blue = 0.802562628052
B_blue = 6.8996252182
C_blue = 33.9425672449

# Base/Max function
x_max_blue = A_blue + C_blue * np.cos(t)
y_max_blue = B_blue + C_blue * np.sin(t)

# MSD1:
x_msd1_blue = A_blue + (C_blue-offset_MSD1_blue) * np.cos(t)
y_msd1_blue = B_blue + (C_blue-offset_MSD1_blue) * np.sin(t)

# MSD2:
x_msd2_blue = A_blue + (C_blue-offset_MSD2_blue) * np.cos(t)
y_msd2_blue = B_blue + (C_blue-offset_MSD2_blue) * np.sin(t)

# ALARM:
x_alarm_blue = A_blue + (C_blue-offset_alarm_blue) * np.cos(t)
y_alarm_blue = B_blue + (C_blue-offset_alarm_blue) * np.sin(t)

# Constants from black circle
A_black = 18.29
B_black = 7.21
C_black = 19.25

# Base functions
x_max_black = A_black + C_black * np.cos(t)
y_max_black = B_black + C_black * np.sin(t)

# MSD1:
x_msd1_black = A_black + (C_black-offset_MSD1_black) * np.cos(t)
y_msd1_black = B_black + (C_black-offset_MSD1_black) * np.sin(t)

# MSD2:
x_msd2_black = A_black + (C_black-offset_MSD2_black) * np.cos(t)
y_msd2_black = B_black + (C_black-offset_MSD2_black) * np.sin(t)

# ALARM:
x_alarm_black = A_black + (C_black-offset_alarm_black) * np.cos(t)
y_alarm_black = B_black + (C_black-offset_alarm_black) * np.sin(t)


# Constants for red circle
A_red = -1.5492
B_red = -36.7419
C_red = 61.0017

# Base functions (renamed for black)
x_max_red = A_red + C_red * np.cos(t)
y_max_red = B_red + C_red * np.sin(t)

# MSD1:
x_msd1_red = A_red + (C_red-offset_MSD1_red) * np.cos(t)
y_msd1_red = B_red + (C_red-offset_MSD1_red) * np.sin(t)

# MSD2:
x_msd2_red = A_red + (C_red-offset_MSD2_red) * np.cos(t)
y_msd2_red = B_red + (C_red-offset_MSD2_red) * np.sin(t)

# ALARM:
x_alarm_red = A_red + (C_red-offset_alarm_red) * np.cos(t)
y_alarm_red = B_red + (C_red-offset_alarm_red) * np.sin(t)

#Grey line - ingress bounds

#Max

#ingress_kanon = 5.5658 * z_flange_dist - 23.864

#if ingress_kanon > ingress_marine:
#  ingress_bound = ingress_kanon
#else:
ingress_bound = ingress_marine

#MSD1
ingress_bound_MSD1 = ingress_bound# + offset_MSD1_ingress

#MSD2
ingress_bound_MSD2 = ingress_bound + 0.5 #+ offset_MSD2_ingress

#ALARM
ingress_bound_alarm = ingress_bound + 3.5 # offset_alarm_ingress


# Reference points
X_ref = [3.93, 12.254, 17.64, 23.997, 34.255, 34.531, 32.133, 19.035, 3.93]
Y_ref = [5.758, 10.656, 21.163, 18.653, 12.647, 3.093, -6.158, -12.025, -5.617]



#compute all intersection points
# Circle intersection function
def circle_intersections(x1, y1, r1, x2, y2, r2):
    dx = x2 - x1
    dy = y2 - y1
    d = np.hypot(dx, dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []  # no intersection
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = np.sqrt(r1**2 - a**2)
    x0 = x1 + a * dx/d
    y0 = y1 + a * dy/d
    rx = -dy * (h/d)
    ry = dx * (h/d)
    intersection1 = (x0 + rx, y0 + ry)
    intersection2 = (x0 - rx, y0 - ry)
    return [intersection1, intersection2]


#intersection of circle and vertical line
def circle_vertical_intersection(A, B, C, k, y_limit=10):
    """
    Intersections of circle (A,B,C) with vertical line x=k,
    only keeping intersections where y <= y_limit.
    """
    inside = C**2 - (k - A)**2

    if inside < 0:
        return []  # no intersection

    y1 = B + np.sqrt(inside)
    y2 = B + -np.sqrt(inside)

    points = [(k, y1), (k, y2)]
    # Keep only those below the y_limit
    return [(x, y) for (x, y) in points if y <= y_limit]

## MAX BOUNDS
# Compute intersection points of circles
points_pink_red = circle_intersections(A_pink, B_pink, C_pink, A_red, B_red, C_red)
points_red_blue = circle_intersections(A_blue, B_blue, C_blue, A_red, B_red, C_red)
# If there are intersections, get the highest y of blue that meets red
if points_red_blue:
    y_max_blue_intersect = max(pt[1] for pt in points_red_blue)
else:
    y_max_blue_intersect = y_max_blue  # fallback
points_blue_black = circle_intersections(A_blue, B_blue, C_blue, A_black, B_black, C_black)
# Filter out intersections where blue is above y=15
points_blue_black = [pt for pt in points_blue_black if pt[1] <= 15]

# #compute intersection points of circle and vertical line - REMOVED INGRESS BOUND
# points_black_grey = circle_vertical_intersection(A_black, B_black, C_black, ingress_bound)
# points_grey_pink = circle_vertical_intersection(A_pink, B_pink, C_pink, ingress_bound)
points_black_pink = circle_intersections(A_black, B_black, C_black, A_pink, B_pink, C_pink)


#MSD1 BOUNDS
# Compute intersection points of circles
points_pink_red_msd1 = circle_intersections(A_pink, B_pink, C_pink-offset_MSD1_pink, A_red, B_red, C_red-offset_MSD1_red)
points_red_blue_msd1 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD1_blue, A_red, B_red, C_red-offset_MSD1_red)
# If there are intersections, get the highest y of blue that meets red
if points_red_blue_msd1:
    y_max_blue_intersect_msd1 = max(pt[1] for pt in points_red_blue_msd1)
else:
    y_max_blue_intersect_msd1 = y_msd1_blue  # fallback
points_blue_black_msd1 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD1_blue, A_black, B_black, C_black-offset_MSD1_black)
# Filter out intersections where blue is above y=15
points_blue_black_msd1 = [pt for pt in points_blue_black_msd1 if pt[1] <= 15]

# #compute intersection points of circle and vertical line - REMOVED VERTICAL INGRESS
# points_black_grey_msd1 = circle_vertical_intersection(A_black, B_black, C_black-offset_MSD1, ingress_bound_MSD1)
# points_grey_pink_msd1 = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_MSD1, ingress_bound_MSD1)
points_black_pink_msd1 = circle_intersections(A_black, B_black, C_black-offset_MSD1_black, A_pink, B_pink, C_pink-offset_MSD1_pink)

# MSD2 BOUNDS
points_pink_red_msd2 = circle_intersections(A_pink, B_pink, C_pink-offset_MSD2_pink,A_red, B_red, C_red-offset_MSD2_red)
points_red_blue_msd2 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD2_blue,A_red, B_red, C_red-offset_MSD2_red)
if points_red_blue_msd2:
    y_max_blue_intersect_msd2 = max(pt[1] for pt in points_red_blue_msd2)
else:
    y_max_blue_intersect_msd2 = y_msd2_blue
points_blue_black_msd2 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD2_blue,A_black, B_black, C_black-offset_MSD2_black)
points_blue_black_msd2 = [pt for pt in points_blue_black_msd2 if pt[1] <= 15]
#REMOVED VERTICAL INGRESS
#points_black_grey_msd2 = circle_vertical_intersection(A_black, B_black, C_black-offset_MSD2, ingress_bound_MSD2)
#points_grey_pink_msd2 = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_MSD2, ingress_bound_MSD2)
points_black_pink_msd2 = circle_intersections(A_black, B_black, C_black-offset_MSD2_black, A_pink, B_pink, C_pink-offset_MSD2_pink)


# ALARM BOUNDS
points_pink_red_alarm = circle_intersections(A_pink, B_pink, C_pink-offset_alarm_pink,A_red, B_red, C_red-offset_alarm_red)
points_red_blue_alarm = circle_intersections(A_blue, B_blue, C_blue-offset_alarm_blue,A_red, B_red, C_red-offset_alarm_red)
if points_red_blue_alarm:
    y_max_blue_intersect_alarm = max(pt[1] for pt in points_red_blue_alarm)
else:
    y_max_blue_intersect_alarm = y_alarm_blue
points_blue_black_alarm = circle_intersections(A_blue, B_blue, C_blue-offset_alarm_blue,A_black, B_black, C_black-offset_alarm_black)
points_blue_black_alarm = [pt for pt in points_blue_black_alarm if pt[1] <= 15]

# REMOVE VERTICAL INGRESS BOUND
# points_black_grey_alarm = circle_vertical_intersection(A_black, B_black, C_black-offset_alarm, ingress_bound_alarm)
# points_grey_pink_alarm = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_alarm, ingress_bound_alarm)
points_black_pink_alarm = circle_intersections(A_black, B_black, C_black-offset_alarm_black, A_pink, B_pink, C_pink-offset_alarm_pink)


#Compute associate angles with intersection points

#MAX
theta_pink_1 = 2*np.pi - np.arccos(np.clip((points_black_pink[0][0] - A_pink)/C_pink, -1, 1))
theta_pink_2 = 2 * np.pi - np.arccos(np.clip((points_pink_red[0][0] - A_pink) / C_pink, -1, 1))

theta_red_1 = np.arccos(np.clip((points_red_blue[0][0] - A_red)/C_red, -1, 1))
theta_red_2 = np.arccos(np.clip((points_pink_red[0][0] - A_red)/C_red, -1, 1))

theta_blue_1 = np.arccos(np.clip((points_red_blue[0][0] - A_blue)/C_blue, -1, 1))
theta_blue_2 = 2*np.pi - np.arccos(np.clip((points_blue_black[0][0] - A_blue)/C_blue, -1, 1))

theta_black_1 = 2*np.pi - np.arccos(np.clip((points_black_pink[0][0] - A_black)/C_black, -1, 1))
theta_black_2 = 2*np.pi - np.arccos(np.clip((points_blue_black[0][0] - A_black)/C_black, -1, 1))

# lower_grey_1 = points_black_grey[0][1]
# upper_grey_2 = points_grey_pink[0][1]

#MSD1
theta_pink_1_msd1 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd1[0][0] - A_pink)/(C_pink-offset_MSD1_pink), -1, 1))
theta_pink_2_msd1 = 2 * np.pi - np.arccos(np.clip((points_pink_red_msd1[0][0] - A_pink) / (C_pink-offset_MSD1_pink), -1, 1))

theta_red_1_msd1 = np.arccos(np.clip((points_red_blue_msd1[0][0] - A_red)/(C_red -offset_MSD1_red), -1, 1))
theta_red_2_msd1 = np.arccos(np.clip((points_pink_red_msd1[0][0] - A_red)/(C_red-offset_MSD1_red), -1, 1))

theta_blue_1_msd1 = np.arccos(np.clip((points_red_blue_msd1[0][0] - A_blue)/(C_blue-offset_MSD1_blue), -1, 1))
theta_blue_2_msd1 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd1[0][0] - A_blue)/(C_blue-offset_MSD1_blue), -1, 1))

theta_black_1_msd1 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd1[0][0] - A_black)/(C_black-offset_MSD1_black), -1, 1))
theta_black_2_msd1 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd1[0][0] - A_black)/(C_black-offset_MSD1_black), -1, 1))

# lower_grey_1_msd1 = points_black_grey_msd1[0][1]
# upper_grey_2_msd1 = points_grey_pink_msd1[0][1-

# MSD2
theta_pink_1_msd2 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd2[0][0] - A_pink)/(C_pink-offset_MSD2_pink), -1, 1))
theta_pink_2_msd2 = 2 * np.pi - np.arccos(np.clip((points_pink_red_msd2[0][0] - A_pink) / (C_pink-offset_MSD2_pink), -1, 1))

theta_red_1_msd2 = np.arccos(np.clip((points_red_blue_msd2[0][0] - A_red)/(C_red-offset_MSD2_red), -1, 1))
theta_red_2_msd2 = np.arccos(np.clip((points_pink_red_msd2[0][0] - A_red)/(C_red-offset_MSD2_red), -1, 1))

theta_blue_1_msd2 = np.arccos(np.clip((points_red_blue_msd2[0][0] - A_blue)/(C_blue-offset_MSD2_blue), -1, 1))
theta_blue_2_msd2 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd2[0][0] - A_blue)/(C_blue-offset_MSD2_blue), -1, 1))

theta_black_1_msd2 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd2[0][0] - A_black)/(C_black-offset_MSD2_black), -1, 1))
theta_black_2_msd2 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd2[0][0] - A_black)/(C_black-offset_MSD2_black), -1, 1))

# lower_grey_1_msd2 = points_black_grey_msd2[0][1]
# upper_grey_2_msd2 = points_grey_pink_msd2[0][1]

# ALARM
theta_pink_1_alarm = 2*np.pi - np.arccos(np.clip((points_black_pink_alarm[0][0] - A_pink)/(C_pink+offset_alarm), -1, 1))
theta_pink_2_alarm = 2 * np.pi - np.arccos(np.clip((points_pink_red_alarm[0][0] - A_pink) / (C_pink+offset_alarm), -1, 1))

theta_red_1_alarm = np.arccos(np.clip((points_red_blue_alarm[0][0] - A_red)/(C_red-offset_alarm), -1, 1))
theta_red_2_alarm = np.arccos(np.clip((points_pink_red_alarm[0][0] - A_red)/(C_red-offset_alarm), -1, 1))

theta_blue_1_alarm = np.arccos(np.clip((points_red_blue_alarm[0][0] - A_blue)/(C_blue-offset_alarm), -1, 1))
theta_blue_2_alarm = 2*np.pi - np.arccos(np.clip((points_blue_black_alarm[0][0] - A_blue)/(C_blue-offset_alarm), -1, 1)) # Removed '+ offset_alarm' from numerator

theta_black_1_alarm = 2*np.pi - np.arccos(np.clip((points_black_pink_alarm[0][0] - A_black)/(C_black-offset_alarm), -1, 1))
theta_black_2_alarm = 2*np.pi - np.arccos(np.clip((points_blue_black_alarm[0][0] - A_black)/(C_black-offset_alarm), -1, 1))

# lower_grey_1_alarm = points_black_grey_alarm[0][1]
# upper_grey_2_alarm = points_grey_pink_alarm[0][1]

#plot arcs within bounds of intersection points

def calculate_arc_only(x_max, y_max, theta_1, theta_2):
    # Angle mask
    if theta_1 < theta_2:
        mask = (t >= theta_1) & (t <= theta_2)
    else:
        mask = (t >= theta_1) | (t <= theta_2)

    # Extract arc points
    x_arc = x_max[mask]
    y_arc = y_max[mask]
    t_arc = t[mask]

    # Fix angle ordering by unwrapping
    t_unwrapped = np.unwrap(t_arc)

    # sort by corrected angle
    order = np.argsort(t_unwrapped)

    x_arc = x_arc[order]
    y_arc = y_arc[order]

    return x_arc, y_arc

fig_orth, ax = plt.subplots(figsize=(8, 7))

#MAX
x_pink_arc, y_pink_arc = calculate_arc_only(x_max_pink, y_max_pink, theta_pink_1, theta_pink_2)
x_red_arc, y_red_arc = calculate_arc_only(x_max_red, y_max_red, theta_red_1, theta_red_2)
x_blue_arc, y_blue_arc = calculate_arc_only(x_max_blue, y_max_blue, theta_blue_2, theta_blue_1)
x_black_arc, y_black_arc = calculate_arc_only(x_max_black, y_max_black, theta_black_1, theta_black_2)
#x_grey = np.array([ingress_bound, ingress_bound])
#y_grey = np.array([lower_grey_1, upper_grey_2])

#plt.plot(x_grey, y_grey, color = 'black', label = 'Failure') #grey
plt.plot(x_pink_arc, y_pink_arc, color='red',linestyle ='dashed', label = "Failure") #pink
plt.plot(x_red_arc, y_red_arc, color='red', linestyle='dashed') #red
plt.plot(x_blue_arc, y_blue_arc, color='red',linestyle='dashed') #blue
plt.plot(x_black_arc, y_black_arc, color='red',linestyle='dashed') #black

# --- MSD1 ---
x_msd1_pink, y_msd1_pink = calculate_arc_only(x_msd1_pink, y_msd1_pink, theta_pink_1_msd1, theta_pink_2_msd1)
x_msd1_red, y_msd1_red = calculate_arc_only(x_msd1_red, y_msd1_red, theta_red_1_msd1, theta_red_2_msd1)
x_msd1_blue, y_msd1_blue = calculate_arc_only(x_msd1_blue, y_msd1_blue, theta_blue_2_msd1, theta_blue_1_msd1)
x_msd1_black, y_msd1_black = calculate_arc_only(x_msd1_black, y_msd1_black, theta_black_1_msd1, theta_black_2_msd1)
#x_msd1_grey = np.array([ingress_bound + offset_MSD1, ingress_bound + offset_MSD1])
#y_msd1_grey = np.array([lower_grey_1_msd1, upper_grey_2_msd1])

#plt.plot(x_msd1_grey, y_msd1_grey, color='green', label = 'MSD1')
plt.plot(x_msd1_pink, y_msd1_pink, color='red', label = 'MSD2')
plt.plot(x_msd1_red, y_msd1_red, color='red')
plt.plot(x_msd1_blue, y_msd1_blue, color='red')
plt.plot(x_msd1_black, y_msd1_black, color='red')

# --- MSD2 ---
x_msd2_pink, y_msd2_pink = calculate_arc_only(x_msd2_pink, y_msd2_pink, theta_pink_1_msd2, theta_pink_2_msd2)
x_msd2_red, y_msd2_red = calculate_arc_only(x_msd2_red, y_msd2_red, theta_red_1_msd2, theta_red_2_msd2)
x_msd2_blue, y_msd2_blue = calculate_arc_only(x_msd2_blue, y_msd2_blue, theta_blue_2_msd2, theta_blue_1_msd2)
x_msd2_black, y_msd2_black = calculate_arc_only(x_msd2_black, y_msd2_black, theta_black_1_msd2, theta_black_2_msd2)
# x_msd2_grey = np.array([ingress_bound + offset_MSD2, ingress_bound + offset_MSD2])
# y_msd2_grey = np.array([lower_grey_1_msd2, upper_grey_2_msd2])

#plt.plot(x_msd2_grey, y_msd2_grey, color='orange', label = 'MSD2')
plt.plot(x_msd2_pink, y_msd2_pink, color='orange', label = 'MSD1')
plt.plot(x_msd2_red, y_msd2_red, color='orange')
plt.plot(x_msd2_blue, y_msd2_blue, color='orange')
plt.plot(x_msd2_black, y_msd2_black, color='orange')

# --- alarm ---
x_alarm_pink, y_alarm_pink = calculate_arc_only(x_alarm_pink, y_alarm_pink, theta_pink_1_alarm, theta_pink_2_alarm)
x_alarm_red, y_alarm_red = calculate_arc_only(x_alarm_red, y_alarm_red, theta_red_1_alarm, theta_red_2_alarm)
x_alarm_blue, y_alarm_blue = calculate_arc_only(x_alarm_blue, y_alarm_blue, theta_blue_2_alarm, theta_blue_1_alarm)
x_alarm_black, y_alarm_black = calculate_arc_only(x_alarm_black, y_alarm_black, theta_black_1_alarm, theta_black_2_alarm)
# x_alarm_grey = np.array([ingress_bound + offset_alarm, ingress_bound + offset_alarm])
# y_alarm_grey = np.array([lower_grey_1_alarm, upper_grey_2_alarm])

# plt.plot(x_alarm_grey, y_alarm_grey, color='purple', label = 'Alarm')
plt.plot(x_alarm_pink, y_alarm_pink, color='purple', label = 'Alarm')
plt.plot(x_alarm_red, y_alarm_red, color='purple')
plt.plot(x_alarm_blue, y_alarm_blue, color='purple')
plt.plot(x_alarm_black, y_alarm_black, color='purple')

plt.scatter(x_origin_B, 0, s=50, marker='D', color = 'grey', label = "MLA B Riser")
plt.scatter(mlab_x, mlab_y, s=50, color = 'blue', label = "Presentation Flange")
plt.plot([x_origin_B, x_origin_B],[0,9], color='black')
plt.plot([x_origin_B, 9],[9,16], color='black')
plt.plot([9,mlab_x-1],[16,mlab_y+2], color='black')
plt.plot([mlab_x-1, mlab_x-1],[mlab_y+2, mlab_y], color='black')
plt.plot([mlab_x-1, mlab_x],[mlab_y, mlab_y], color='black')


#plot doc edge
plt.axvline(x=0, color='black', linewidth=1, label = 'Doc Edge')

plt.xlabel("Reach")
plt.ylabel("Y-Axis")
plt.title("91-MLA-9901B Envelope (R-Y)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout() # Adjust layout to prevent legend overlap
plt.show()

col_2, col_3, = st.columns(2)

with col_2:
  st.pyplot(fig_orth)


with col_3:

  fig_slew, ax = plt.subplots(figsize=(8, 7))

  r = inboard_length + outboard_length + x_tsa_extension

  theta_main_max = 33 * np.pi / 180

    # Thresholds (angle_deg, ingress_m, color, label)
  slew_thresholds = [
      (31, ingress_bound_MSD1, "red", "MSD2"),
      (30, ingress_bound_MSD2, "orange", "MSD1"),
      (29, ingress_bound_alarm, "purple", "Alarm")]

  # -----------------------------
  # Functions
  # -----------------------------
  def arc_points(x_origin, z_origin, r, theta_max, num=300):
      theta = np.linspace(-theta_max, theta_max, num)
      x = x_origin + r * np.cos(theta)
      z = z_origin + r * np.sin(theta)
      return x, z

  def slew_line(x_origin, z_origin, theta, x_start, r, num=200):
      s_start = (x_start - x_origin) / np.cos(theta)
      s = np.linspace(s_start, r, num)
      x = x_origin + s * np.cos(theta)
      z = z_origin + s * np.sin(theta)
      return x, z

  def intersection_with_vertical(x_vert, x_origin, z_origin, theta):
      z = np.tan(theta) * (x_vert - x_origin) + z_origin
      return x_vert, z

  # -----------------------------
  # Plot
  # -----------------------------

  # Main arc ±33° (grey dashed)
  x_arc, z_arc = arc_points(x_origin_B, z_origin_B, r, theta_main_max)
  plt.plot(x_arc, z_arc, color="grey", linestyle="--", linewidth=2)

  # Main slew lines ±33° (black = Failure)
  for theta in [theta_main_max, -theta_main_max]:
      x_slew, z_slew = slew_line(x_origin_B, z_origin_B, theta, ingress_bound, r)
      plt.plot(x_slew, z_slew, color="red", linestyle='dashed', label="Failure" if theta>0 else None)

  # Threshold slew lines with matched ingress, color, and label
  for angle_deg, ingress, color, label in slew_thresholds:
      theta_th = np.deg2rad(angle_deg)
      for i, theta in enumerate([theta_th, -theta_th]):
          x_slew, z_slew = slew_line(x_origin_B, z_origin_B, theta, ingress, r)
          plt.plot(x_slew, z_slew, color=color, linewidth=2,
                  label=label if i==0 else None)  # label only for positive angle

  # Ingress bounds (cut at intersection with respective slews)
  all_ingress = [ingress_bound] + [ing for _, ing, _, _ in slew_thresholds]
  all_slew_angles = [theta_main_max] + [np.deg2rad(angle) for angle, _, _, _ in slew_thresholds]
  all_colors = ["red"] + [color for _, _, color, _ in slew_thresholds]

  for x_ing, theta_up, color in zip(all_ingress, all_slew_angles, all_colors):
      theta_low = -theta_up
      z_upper = intersection_with_vertical(x_ing, x_origin_B, z_origin_B, theta_up)[1]
      z_lower = intersection_with_vertical(x_ing, x_origin_B, z_origin_B, theta_low)[1]
      plt.plot([x_ing, x_ing], [z_lower, z_upper], color=color, linewidth=2,
              )

  # MLA Riser Location
  plt.scatter(x_origin_B, z_origin_B,  s=50, marker='D', color = 'Grey', label="MLA B Riser")
  plt.scatter(mlab_x, mlab_z, s=50, color = 'blue', label = "Presentation Flange")
  plt.plot([x_origin_B,mlab_x], [z_origin_B,mlab_z], color='black')

  #plot doc edge
  plt.axvline(x=0, color='black', linewidth=1, label = 'Doc Edge')

  plt.axis("equal")
  plt.grid(True)
  plt.xlabel("X-Axis")
  plt.ylabel("Z-Axis")
  ax.invert_yaxis()
  plt.title("91-MLA-9901B Ingress (X-Z)")
  plt.legend()
  plt.show()

  st.pyplot(fig_slew)



st.markdown(
    "<h3 style='text-align: center;'>BERTH 1 91-MLA-9901C MARINE LOADING ARM POSITIONING SYSTEM 2 BUTANE</h3>",
    unsafe_allow_html=True
)

inboard_length = 17500/1000    #length of the inboard arm in m
riser_length = 9000/1000       #length of the MLA riser in m
outboard_length = 19250/1000   #length of the outboard arm in m

x_tsa_extension = 1040/1000
y_tsa_extension = 2185/1000
z_tsa_extension_B= 0.6192  # +0.6192 for MLA B, -0.6192 for MLA C
z_tsa_extension_C = -0.6192

x_CD_ext = .254

ingress_marine = 15.75
z_flange_dist = 2.25

#MLA & gangway origins
x_origin_B = + -3.5
z_origin_B = -2.500

x_origin_C = -3.5
z_origin_C = +2.500

offset_alarm = 7  #offset applied to acheive alarm function
offset_MSD2 = 2     #offset applied to acheive MSD2 function
offset_MSD1 = 1.5   #offset applied to acheive MSD1 function

#IF OFFSET VARRIES BY CIRCLE/LINE, INPUT DESIRED OFFSET FOR EACH BELOW
#pink offsets
offset_alarm_pink = -offset_alarm
offset_MSD2_pink = -offset_MSD2
offset_MSD1_pink = -offset_MSD1

#red offsets
offset_alarm_red = offset_alarm
offset_MSD2_red = offset_MSD2
offset_MSD1_red = offset_MSD1

#blue offsets
offset_alarm_blue = offset_alarm
offset_MSD2_blue = offset_MSD2
offset_MSD1_blue = offset_MSD1

#black offsets
offset_alarm_black = offset_alarm
offset_MSD2_black = offset_MSD2
offset_MSD1_black = offset_MSD1

#grey offsets
offset_alarm_ingress = offset_alarm
offset_MSD2_ingress = offset_MSD2
offset_MSD1_ingress = offset_MSD1

#equtions for circles
# Array t from 0 to 2*pi
t = np.linspace(0, 2*np.pi, 10000)

# Core function of all circles is (x+A^2) + (y+B^2) = (D +C) edit ithis

# Constants from pink circle
A_pink = -1.358533
B_pink = 24.26789
C_pink = 19.25250575

# Base functions
x_max_pink = A_pink + C_pink * np.cos(t)
y_max_pink = B_pink + C_pink * np.sin(t)

# MSD1:
x_msd1_pink = A_pink + (C_pink-offset_MSD1_pink) * np.cos(t)
y_msd1_pink = B_pink + (C_pink-offset_MSD1_pink) * np.sin(t)

# MSD2:
x_msd2_pink = A_pink + (C_pink-offset_MSD2_pink) * np.cos(t)
y_msd2_pink = B_pink + (C_pink-offset_MSD2_pink) * np.sin(t)

# ALARM:
x_alarm_pink = A_pink + (C_pink-offset_alarm_pink) * np.cos(t)
y_alarm_pink = B_pink + (C_pink-offset_alarm_pink) * np.sin(t)

# Constants for blue circle
A_blue = 0.802562628052
B_blue = 6.8996252182
C_blue = 33.9425672449

# Base/Max function
x_max_blue = A_blue + C_blue * np.cos(t)
y_max_blue = B_blue + C_blue * np.sin(t)

# MSD1:
x_msd1_blue = A_blue + (C_blue-offset_MSD1_blue) * np.cos(t)
y_msd1_blue = B_blue + (C_blue-offset_MSD1_blue) * np.sin(t)

# MSD2:
x_msd2_blue = A_blue + (C_blue-offset_MSD2_blue) * np.cos(t)
y_msd2_blue = B_blue + (C_blue-offset_MSD2_blue) * np.sin(t)

# ALARM:
x_alarm_blue = A_blue + (C_blue-offset_alarm_blue) * np.cos(t)
y_alarm_blue = B_blue + (C_blue-offset_alarm_blue) * np.sin(t)

# Constants from black circle
A_black = 18.29
B_black = 7.21
C_black = 19.25

# Base functions
x_max_black = A_black + C_black * np.cos(t)
y_max_black = B_black + C_black * np.sin(t)

# MSD1:
x_msd1_black = A_black + (C_black-offset_MSD1_black) * np.cos(t)
y_msd1_black = B_black + (C_black-offset_MSD1_black) * np.sin(t)

# MSD2:
x_msd2_black = A_black + (C_black-offset_MSD2_black) * np.cos(t)
y_msd2_black = B_black + (C_black-offset_MSD2_black) * np.sin(t)

# ALARM:
x_alarm_black = A_black + (C_black-offset_alarm_black) * np.cos(t)
y_alarm_black = B_black + (C_black-offset_alarm_black) * np.sin(t)


# Constants for red circle
A_red = -1.5492
B_red = -36.7419
C_red = 61.0017

# Base functions (renamed for black)
x_max_red = A_red + C_red * np.cos(t)
y_max_red = B_red + C_red * np.sin(t)

# MSD1:
x_msd1_red = A_red + (C_red-offset_MSD1_red) * np.cos(t)
y_msd1_red = B_red + (C_red-offset_MSD1_red) * np.sin(t)

# MSD2:
x_msd2_red = A_red + (C_red-offset_MSD2_red) * np.cos(t)
y_msd2_red = B_red + (C_red-offset_MSD2_red) * np.sin(t)

# ALARM:
x_alarm_red = A_red + (C_red-offset_alarm_red) * np.cos(t)
y_alarm_red = B_red + (C_red-offset_alarm_red) * np.sin(t)

#Grey line - ingress bounds

#Max

#ingress_kanon = 5.5658 * z_flange_dist - 23.864

#if ingress_kanon > ingress_marine:
#  ingress_bound = ingress_kanon
#else:
ingress_bound = ingress_marine

#MSD1
ingress_bound_MSD1 = ingress_bound #+ offset_MSD1_ingress

#MSD2
ingress_bound_MSD2 = ingress_bound + 0.5 #+ offset_MSD2_ingress

#ALARM
ingress_bound_alarm = ingress_bound + 3.5 # offset_alarm_ingress


#compute all intersection points
# Circle intersection function
def circle_intersections(x1, y1, r1, x2, y2, r2):
    dx = x2 - x1
    dy = y2 - y1
    d = np.hypot(dx, dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []  # no intersection
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = np.sqrt(r1**2 - a**2)
    x0 = x1 + a * dx/d
    y0 = y1 + a * dy/d
    rx = -dy * (h/d)
    ry = dx * (h/d)
    intersection1 = (x0 + rx, y0 + ry)
    intersection2 = (x0 - rx, y0 - ry)
    return [intersection1, intersection2]


#intersection of circle and vertical line
def circle_vertical_intersection(A, B, C, k, y_limit=10):
    """
    Intersections of circle (A,B,C) with vertical line x=k,
    only keeping intersections where y <= y_limit.
    """
    inside = C**2 - (k - A)**2

    if inside < 0:
        return []  # no intersection

    y1 = B + np.sqrt(inside)
    y2 = B + -np.sqrt(inside)

    points = [(k, y1), (k, y2)]
    # Keep only those below the y_limit
    return [(x, y) for (x, y) in points if y <= y_limit]

## MAX BOUNDS
# Compute intersection points of circles
points_pink_red = circle_intersections(A_pink, B_pink, C_pink, A_red, B_red, C_red)
points_red_blue = circle_intersections(A_blue, B_blue, C_blue, A_red, B_red, C_red)
# If there are intersections, get the highest y of blue that meets red
if points_red_blue:
    y_max_blue_intersect = max(pt[1] for pt in points_red_blue)
else:
    y_max_blue_intersect = y_max_blue  # fallback
points_blue_black = circle_intersections(A_blue, B_blue, C_blue, A_black, B_black, C_black)
# Filter out intersections where blue is above y=15
points_blue_black = [pt for pt in points_blue_black if pt[1] <= 15]

# #compute intersection points of circle and vertical line - REMOVED INGRESS BOUND
# points_black_grey = circle_vertical_intersection(A_black, B_black, C_black, ingress_bound)
# points_grey_pink = circle_vertical_intersection(A_pink, B_pink, C_pink, ingress_bound)
points_black_pink = circle_intersections(A_black, B_black, C_black, A_pink, B_pink, C_pink)


#MSD1 BOUNDS
# Compute intersection points of circles
points_pink_red_msd1 = circle_intersections(A_pink, B_pink, C_pink-offset_MSD1_pink, A_red, B_red, C_red-offset_MSD1_red)
points_red_blue_msd1 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD1_blue, A_red, B_red, C_red-offset_MSD1_red)
# If there are intersections, get the highest y of blue that meets red
if points_red_blue_msd1:
    y_max_blue_intersect_msd1 = max(pt[1] for pt in points_red_blue_msd1)
else:
    y_max_blue_intersect_msd1 = y_msd1_blue  # fallback
points_blue_black_msd1 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD1_blue, A_black, B_black, C_black-offset_MSD1_black)
# Filter out intersections where blue is above y=15
points_blue_black_msd1 = [pt for pt in points_blue_black_msd1 if pt[1] <= 15]

# #compute intersection points of circle and vertical line - REMOVED VERTICAL INGRESS
# points_black_grey_msd1 = circle_vertical_intersection(A_black, B_black, C_black-offset_MSD1, ingress_bound_MSD1)
# points_grey_pink_msd1 = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_MSD1, ingress_bound_MSD1)
points_black_pink_msd1 = circle_intersections(A_black, B_black, C_black-offset_MSD1_black, A_pink, B_pink, C_pink-offset_MSD1_pink)

# MSD2 BOUNDS
points_pink_red_msd2 = circle_intersections(A_pink, B_pink, C_pink-offset_MSD2_pink,A_red, B_red, C_red-offset_MSD2_red)
points_red_blue_msd2 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD2_blue,A_red, B_red, C_red-offset_MSD2_red)
if points_red_blue_msd2:
    y_max_blue_intersect_msd2 = max(pt[1] for pt in points_red_blue_msd2)
else:
    y_max_blue_intersect_msd2 = y_msd2_blue
points_blue_black_msd2 = circle_intersections(A_blue, B_blue, C_blue-offset_MSD2_blue,A_black, B_black, C_black-offset_MSD2_black)
points_blue_black_msd2 = [pt for pt in points_blue_black_msd2 if pt[1] <= 15]
#REMOVED VERTICAL INGRESS
#points_black_grey_msd2 = circle_vertical_intersection(A_black, B_black, C_black-offset_MSD2, ingress_bound_MSD2)
#points_grey_pink_msd2 = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_MSD2, ingress_bound_MSD2)
points_black_pink_msd2 = circle_intersections(A_black, B_black, C_black-offset_MSD2_black, A_pink, B_pink, C_pink-offset_MSD2_pink)


# ALARM BOUNDS
points_pink_red_alarm = circle_intersections(A_pink, B_pink, C_pink-offset_alarm_pink,A_red, B_red, C_red-offset_alarm_red)
points_red_blue_alarm = circle_intersections(A_blue, B_blue, C_blue-offset_alarm_blue,A_red, B_red, C_red-offset_alarm_red)
if points_red_blue_alarm:
    y_max_blue_intersect_alarm = max(pt[1] for pt in points_red_blue_alarm)
else:
    y_max_blue_intersect_alarm = y_alarm_blue
points_blue_black_alarm = circle_intersections(A_blue, B_blue, C_blue-offset_alarm_blue,A_black, B_black, C_black-offset_alarm_black)
points_blue_black_alarm = [pt for pt in points_blue_black_alarm if pt[1] <= 15]

# REMOVE VERTICAL INGRESS BOUND
# points_black_grey_alarm = circle_vertical_intersection(A_black, B_black, C_black-offset_alarm, ingress_bound_alarm)
# points_grey_pink_alarm = circle_vertical_intersection(A_pink, B_pink, C_pink+offset_alarm, ingress_bound_alarm)
points_black_pink_alarm = circle_intersections(A_black, B_black, C_black-offset_alarm_black, A_pink, B_pink, C_pink-offset_alarm_pink)


#Compute associate angles with intersection points

#MAX
theta_pink_1 = 2*np.pi - np.arccos(np.clip((points_black_pink[0][0] - A_pink)/C_pink, -1, 1))
theta_pink_2 = 2 * np.pi - np.arccos(np.clip((points_pink_red[0][0] - A_pink) / C_pink, -1, 1))

theta_red_1 = np.arccos(np.clip((points_red_blue[0][0] - A_red)/C_red, -1, 1))
theta_red_2 = np.arccos(np.clip((points_pink_red[0][0] - A_red)/C_red, -1, 1))

theta_blue_1 = np.arccos(np.clip((points_red_blue[0][0] - A_blue)/C_blue, -1, 1))
theta_blue_2 = 2*np.pi - np.arccos(np.clip((points_blue_black[0][0] - A_blue)/C_blue, -1, 1))

theta_black_1 = 2*np.pi - np.arccos(np.clip((points_black_pink[0][0] - A_black)/C_black, -1, 1))
theta_black_2 = 2*np.pi - np.arccos(np.clip((points_blue_black[0][0] - A_black)/C_black, -1, 1))

# lower_grey_1 = points_black_grey[0][1]
# upper_grey_2 = points_grey_pink[0][1]

#MSD1
theta_pink_1_msd1 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd1[0][0] - A_pink)/(C_pink-offset_MSD1_pink), -1, 1))
theta_pink_2_msd1 = 2 * np.pi - np.arccos(np.clip((points_pink_red_msd1[0][0] - A_pink) / (C_pink-offset_MSD1_pink), -1, 1))

theta_red_1_msd1 = np.arccos(np.clip((points_red_blue_msd1[0][0] - A_red)/(C_red -offset_MSD1_red), -1, 1))
theta_red_2_msd1 = np.arccos(np.clip((points_pink_red_msd1[0][0] - A_red)/(C_red-offset_MSD1_red), -1, 1))

theta_blue_1_msd1 = np.arccos(np.clip((points_red_blue_msd1[0][0] - A_blue)/(C_blue-offset_MSD1_blue), -1, 1))
theta_blue_2_msd1 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd1[0][0] - A_blue)/(C_blue-offset_MSD1_blue), -1, 1))

theta_black_1_msd1 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd1[0][0] - A_black)/(C_black-offset_MSD1_black), -1, 1))
theta_black_2_msd1 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd1[0][0] - A_black)/(C_black-offset_MSD1_black), -1, 1))

# lower_grey_1_msd1 = points_black_grey_msd1[0][1]
# upper_grey_2_msd1 = points_grey_pink_msd1[0][1-

# MSD2
theta_pink_1_msd2 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd2[0][0] - A_pink)/(C_pink-offset_MSD2_pink), -1, 1))
theta_pink_2_msd2 = 2 * np.pi - np.arccos(np.clip((points_pink_red_msd2[0][0] - A_pink) / (C_pink-offset_MSD2_pink), -1, 1))

theta_red_1_msd2 = np.arccos(np.clip((points_red_blue_msd2[0][0] - A_red)/(C_red-offset_MSD2_red), -1, 1))
theta_red_2_msd2 = np.arccos(np.clip((points_pink_red_msd2[0][0] - A_red)/(C_red-offset_MSD2_red), -1, 1))

theta_blue_1_msd2 = np.arccos(np.clip((points_red_blue_msd2[0][0] - A_blue)/(C_blue-offset_MSD2_blue), -1, 1))
theta_blue_2_msd2 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd2[0][0] - A_blue)/(C_blue-offset_MSD2_blue), -1, 1))

theta_black_1_msd2 = 2*np.pi - np.arccos(np.clip((points_black_pink_msd2[0][0] - A_black)/(C_black-offset_MSD2_black), -1, 1))
theta_black_2_msd2 = 2*np.pi - np.arccos(np.clip((points_blue_black_msd2[0][0] - A_black)/(C_black-offset_MSD2_black), -1, 1))

# lower_grey_1_msd2 = points_black_grey_msd2[0][1]
# upper_grey_2_msd2 = points_grey_pink_msd2[0][1]

# ALARM
theta_pink_1_alarm = 2*np.pi - np.arccos(np.clip((points_black_pink_alarm[0][0] - A_pink)/(C_pink+offset_alarm), -1, 1))
theta_pink_2_alarm = 2 * np.pi - np.arccos(np.clip((points_pink_red_alarm[0][0] - A_pink) / (C_pink+offset_alarm), -1, 1))

theta_red_1_alarm = np.arccos(np.clip((points_red_blue_alarm[0][0] - A_red)/(C_red-offset_alarm), -1, 1))
theta_red_2_alarm = np.arccos(np.clip((points_pink_red_alarm[0][0] - A_red)/(C_red-offset_alarm), -1, 1))

theta_blue_1_alarm = np.arccos(np.clip((points_red_blue_alarm[0][0] - A_blue)/(C_blue-offset_alarm), -1, 1))
theta_blue_2_alarm = 2*np.pi - np.arccos(np.clip((points_blue_black_alarm[0][0] - A_blue)/(C_blue-offset_alarm), -1, 1)) # Removed '+ offset_alarm' from numerator

theta_black_1_alarm = 2*np.pi - np.arccos(np.clip((points_black_pink_alarm[0][0] - A_black)/(C_black-offset_alarm), -1, 1))
theta_black_2_alarm = 2*np.pi - np.arccos(np.clip((points_blue_black_alarm[0][0] - A_black)/(C_black-offset_alarm), -1, 1))

# lower_grey_1_alarm = points_black_grey_alarm[0][1]
# upper_grey_2_alarm = points_grey_pink_alarm[0][1]

#plot arcs within bounds of intersection points

def calculate_arc_only(x_max, y_max, theta_1, theta_2):
    # Angle mask
    if theta_1 < theta_2:
        mask = (t >= theta_1) & (t <= theta_2)
    else:
        mask = (t >= theta_1) | (t <= theta_2)

    # Extract arc points
    x_arc = x_max[mask]
    y_arc = y_max[mask]
    t_arc = t[mask]

    # Fix angle ordering by unwrapping
    t_unwrapped = np.unwrap(t_arc)

    # sort by corrected angle
    order = np.argsort(t_unwrapped)

    x_arc = x_arc[order]
    y_arc = y_arc[order]

    return x_arc, y_arc

fig_orth, ax = plt.subplots(figsize=(8, 7))

#MAX
x_pink_arc, y_pink_arc = calculate_arc_only(x_max_pink, y_max_pink, theta_pink_1, theta_pink_2)
x_red_arc, y_red_arc = calculate_arc_only(x_max_red, y_max_red, theta_red_1, theta_red_2)
x_blue_arc, y_blue_arc = calculate_arc_only(x_max_blue, y_max_blue, theta_blue_2, theta_blue_1)
x_black_arc, y_black_arc = calculate_arc_only(x_max_black, y_max_black, theta_black_1, theta_black_2)
#x_grey = np.array([ingress_bound, ingress_bound])
#y_grey = np.array([lower_grey_1, upper_grey_2])

#plt.plot(x_grey, y_grey, color = 'black', label = 'Failure') #grey
plt.plot(x_pink_arc, y_pink_arc, color='red', linestyle ='dashed', label = "Failure") #pink
plt.plot(x_red_arc, y_red_arc, color='red',linestyle ='dashed') #red
plt.plot(x_blue_arc, y_blue_arc, color='red', linestyle ='dashed') #blue
plt.plot(x_black_arc, y_black_arc, color='red', linestyle ='dashed') #black

# --- MSD1 ---
x_msd1_pink, y_msd1_pink = calculate_arc_only(x_msd1_pink, y_msd1_pink, theta_pink_1_msd1, theta_pink_2_msd1)
x_msd1_red, y_msd1_red = calculate_arc_only(x_msd1_red, y_msd1_red, theta_red_1_msd1, theta_red_2_msd1)
x_msd1_blue, y_msd1_blue = calculate_arc_only(x_msd1_blue, y_msd1_blue, theta_blue_2_msd1, theta_blue_1_msd1)
x_msd1_black, y_msd1_black = calculate_arc_only(x_msd1_black, y_msd1_black, theta_black_1_msd1, theta_black_2_msd1)
#x_msd1_grey = np.array([ingress_bound + offset_MSD1, ingress_bound + offset_MSD1])
#y_msd1_grey = np.array([lower_grey_1_msd1, upper_grey_2_msd1])

#plt.plot(x_msd1_grey, y_msd1_grey, color='green', label = 'MSD1')
plt.plot(x_msd1_pink, y_msd1_pink, color='red', label = 'MSD2')
plt.plot(x_msd1_red, y_msd1_red, color='red')
plt.plot(x_msd1_blue, y_msd1_blue, color='red')
plt.plot(x_msd1_black, y_msd1_black, color='red')

# --- MSD2 ---
x_msd2_pink, y_msd2_pink = calculate_arc_only(x_msd2_pink, y_msd2_pink, theta_pink_1_msd2, theta_pink_2_msd2)
x_msd2_red, y_msd2_red = calculate_arc_only(x_msd2_red, y_msd2_red, theta_red_1_msd2, theta_red_2_msd2)
x_msd2_blue, y_msd2_blue = calculate_arc_only(x_msd2_blue, y_msd2_blue, theta_blue_2_msd2, theta_blue_1_msd2)
x_msd2_black, y_msd2_black = calculate_arc_only(x_msd2_black, y_msd2_black, theta_black_1_msd2, theta_black_2_msd2)
# x_msd2_grey = np.array([ingress_bound + offset_MSD2, ingress_bound + offset_MSD2])
# y_msd2_grey = np.array([lower_grey_1_msd2, upper_grey_2_msd2])

#plt.plot(x_msd2_grey, y_msd2_grey, color='orange', label = 'MSD2')
plt.plot(x_msd2_pink, y_msd2_pink, color='orange', label = 'MSD1')
plt.plot(x_msd2_red, y_msd2_red, color='orange')
plt.plot(x_msd2_blue, y_msd2_blue, color='orange')
plt.plot(x_msd2_black, y_msd2_black, color='orange')

# --- alarm ---
x_alarm_pink, y_alarm_pink = calculate_arc_only(x_alarm_pink, y_alarm_pink, theta_pink_1_alarm, theta_pink_2_alarm)
x_alarm_red, y_alarm_red = calculate_arc_only(x_alarm_red, y_alarm_red, theta_red_1_alarm, theta_red_2_alarm)
x_alarm_blue, y_alarm_blue = calculate_arc_only(x_alarm_blue, y_alarm_blue, theta_blue_2_alarm, theta_blue_1_alarm)
x_alarm_black, y_alarm_black = calculate_arc_only(x_alarm_black, y_alarm_black, theta_black_1_alarm, theta_black_2_alarm)
# x_alarm_grey = np.array([ingress_bound + offset_alarm, ingress_bound + offset_alarm])
# y_alarm_grey = np.array([lower_grey_1_alarm, upper_grey_2_alarm])

# plt.plot(x_alarm_grey, y_alarm_grey, color='purple', label = 'Alarm')
plt.plot(x_alarm_pink, y_alarm_pink, color='purple', label = 'Alarm')
plt.plot(x_alarm_red, y_alarm_red, color='purple')
plt.plot(x_alarm_blue, y_alarm_blue, color='purple')
plt.plot(x_alarm_black, y_alarm_black, color='purple')

plt.scatter(x_origin_C, 0, s=50, marker='D', color = 'grey', label = "MLA C Riser")
plt.scatter(mlac_x, mlac_y, s=50, color = 'blue', label = "Presentation Flange")
plt.plot([x_origin_C, x_origin_C],[0,9], color='black')
plt.plot([x_origin_C, 9],[9,16],color='black')
plt.plot([9,mlac_x-1],[16,mlac_y+2], color='black')
plt.plot([mlac_x-1, mlac_x-1],[mlac_y+2, mlac_y], color='black')
plt.plot([mlac_x-1, mlac_x],[mlac_y, mlac_y], color='black')

#plot doc edge
plt.axvline(x=0, color='black', linewidth=1, label = 'Doc Edge')

plt.xlabel("Reach")
plt.ylabel("Y-Axis")
plt.title("91-MLA-9901C Envelope (R-Y)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout() # Adjust layout to prevent legend overlap
plt.show()

col_2, col_3, = st.columns(2)

with col_2:
  st.pyplot(fig_orth)


with col_3:

  fig_slew, ax = plt.subplots(figsize=(8, 7))

  r = inboard_length + outboard_length + x_tsa_extension

  theta_main_max = 33 * np.pi / 180

    # Thresholds (angle_deg, ingress_m, color, label)
  slew_thresholds = [
      (31, ingress_bound_MSD1, "red", "MSD2"),
      (30, ingress_bound_MSD2, "orange", "MSD1"),
      (29, ingress_bound_alarm, "purple", "Alarm")]

  # -----------------------------
  # Functions
  # -----------------------------
  def arc_points(x_origin, z_origin, r, theta_max, num=300):
      theta = np.linspace(-theta_max, theta_max, num)
      x = x_origin + r * np.cos(theta)
      z = z_origin + r * np.sin(theta)
      return x, z

  def slew_line(x_origin, z_origin, theta, x_start, r, num=200):
      s_start = (x_start - x_origin) / np.cos(theta)
      s = np.linspace(s_start, r, num)
      x = x_origin + s * np.cos(theta)
      z = z_origin + s * np.sin(theta)
      return x, z

  def intersection_with_vertical(x_vert, x_origin, z_origin, theta):
      z = np.tan(theta) * (x_vert - x_origin) + z_origin
      return x_vert, z

  # -----------------------------
  # Plot
  # -----------------------------

  # Main arc ±33° (grey dashed)
  x_arc, z_arc = arc_points(x_origin_B, z_origin_B, r, theta_main_max)
  plt.plot(x_arc, z_arc, color="grey", linestyle="--", linewidth=2)

  # Main slew lines ±33° (black = Failure)
  for theta in [theta_main_max, -theta_main_max]:
      x_slew, z_slew = slew_line(x_origin_C, z_origin_B, theta, ingress_bound, r)
      plt.plot(x_slew, z_slew, color="red", linestyle='dashed', label="Failure" if theta>0 else None)

  # Threshold slew lines with matched ingress, color, and label
  for angle_deg, ingress, color, label in slew_thresholds:
      theta_th = np.deg2rad(angle_deg)
      for i, theta in enumerate([theta_th, -theta_th]):
          x_slew, z_slew = slew_line(x_origin_B, z_origin_B, theta, ingress, r)
          plt.plot(x_slew, z_slew, color=color, linewidth=2,
                  label=label if i==0 else None)  # label only for positive angle

  # Ingress bounds (cut at intersection with respective slews)
  all_ingress = [ingress_bound] + [ing for _, ing, _, _ in slew_thresholds]
  all_slew_angles = [theta_main_max] + [np.deg2rad(angle) for angle, _, _, _ in slew_thresholds]
  all_colors = ["red"] + [color for _, _, color, _ in slew_thresholds]

  for x_ing, theta_up, color in zip(all_ingress, all_slew_angles, all_colors):
      theta_low = -theta_up
      z_upper = intersection_with_vertical(x_ing, x_origin_B, z_origin_B, theta_up)[1]
      z_lower = intersection_with_vertical(x_ing, x_origin_B, z_origin_B, theta_low)[1]
      plt.plot([x_ing, x_ing], [z_lower, z_upper], color=color, linewidth=2,
              )

  # MLA Riser Location
  plt.scatter(x_origin_C, z_origin_C,  s=50, marker='D', color = 'Grey', label="MLA C Riser")
  plt.scatter(mlac_x, mlac_z, s=50, color = 'blue', label = "Presentation Flange")
  plt.plot([x_origin_C,mlac_x], [z_origin_C,mlac_z], color='black')

  #plot doc edge
  plt.axvline(x=0, color='black', linewidth=1, label = 'Doc Edge')

  plt.axis("equal")
  plt.grid(True)
  plt.xlabel("X-Axis")
  plt.ylabel("Z-Axis")
  ax.invert_yaxis()
  plt.title("91-MLA-9901C Ingress (X-Z)")
  plt.legend()
  plt.show()

  st.pyplot(fig_slew)

show_overlay_ingress = st.sidebar.checkbox("Show Overlay Ingress Plot")

with col_2:
  if show_overlay_ingress:

      st.header("Overlay Ingress Plot (MLAB + MLAC)")

      fig_overlay, ax = plt.subplots(figsize=(8, 7))

      # -----------------------------
      # Helper functions (reuse)
      # -----------------------------
      def arc_points(x_origin, z_origin, r, theta_max, num=300):
          theta = np.linspace(-theta_max, theta_max, num)
          x = x_origin + r * np.cos(theta)
          z = z_origin + r * np.sin(theta)
          return x, z

      def slew_line(x_origin, z_origin, theta, x_start, r, num=200):
          s_start = (x_start - x_origin) / np.cos(theta)
          s = np.linspace(s_start, r, num)
          x = x_origin + s * np.cos(theta)
          z = z_origin + s * np.sin(theta)
          return x, z

      # -----------------------------
      # Plot BOTH MLA B and MLA C
      # -----------------------------

      systems = [
          ("MLAB", x_origin_B, z_origin_B, mlab_x, mlab_z, "blue"),
          ("MLAC", x_origin_C, z_origin_C, mlac_x, mlac_z, "green"),
      ]

      for name, x_origin, z_origin, flange_x, flange_z, color in systems:

          # Arc
          x_arc, z_arc = arc_points(x_origin, z_origin, r, theta_main_max)
          ax.plot(x_arc, z_arc, linestyle="--", linewidth=1.5, label=f"{name} Arc")

          # Slew limits (±33°)
          for theta in [theta_main_max, -theta_main_max]:
              x_slew, z_slew = slew_line(x_origin, z_origin, theta, ingress_bound, r)
              ax.plot(x_slew, z_slew, linestyle="dashed", color=color,
                      label=f"{name} Failure" if theta > 0 else None)

          # Thresholds
          for angle_deg, ingress, th_color, label in slew_thresholds:
              theta_th = np.deg2rad(angle_deg)
              for i, theta in enumerate([theta_th, -theta_th]):
                  x_slew, z_slew = slew_line(x_origin, z_origin, theta, ingress, r)
                  ax.plot(x_slew, z_slew, color=th_color, linewidth=1.5,
                          label=f"{name} {label}" if i == 0 else None)

          # Flange + riser
          ax.scatter(x_origin, z_origin, marker='D', s=40, label=f"{name} Riser")
          ax.scatter(flange_x, flange_z, s=40, label=f"{name} Flange")
          ax.plot([x_origin, flange_x], [z_origin, flange_z], color=color)

      # Dock edge
      ax.axvline(x=0, color='black', linewidth=1, label='Dock Edge')

      # Flip vertical axis (your requirement)
      ax.invert_yaxis()

      ax.set_aspect("equal")
      ax.grid(True)
      ax.set_xlabel("X-Axis")
      ax.set_ylabel("Z-Axis (flipped)")
      ax.set_title("Overlay Ingress Plot")

      ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

      st.pyplot(fig_overlay)
      plt.close(fig_overlay)


st.header("MLA Positions at Selected Time")

col1, col2 = st.columns(2)

with col1:
    st.subheader("MLAB")
    st.write(f"X: {mlab_x:.3f} m")
    st.write(f"Y: {mlab_y:.3f} m")
    st.write(f"Z: {mlab_z:.3f} m")

with col2:
    st.subheader("MLAC")
    st.write(f"X: {mlac_x:.3f} m")
    st.write(f"Y: {mlac_y:.3f} m")
    st.write(f"Z: {mlac_z:.3f} m")

st.markdown("--- App End ---")
