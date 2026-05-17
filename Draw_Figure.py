import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def draw_train_data():
    # Read CSV file
    data = pd.read_csv('.\Data\D3QN_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time-wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=16)
    # plt.ylim(top=0.0178)
    # Set chart title and axis labels
    plt.title('Train - Final Error vs Episode - D3QN', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)
    plt.legend(loc='upper right')

    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\ReEs_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\ReEs_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=16)
    # plt.ylim(top=0.0178)
    # Set chart title and axis labels
    plt.title('Train - Final Error vs Episode - ReEs', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)
    plt.legend(loc='upper right')

    # Show chart
    plt.show()


    # Read CSV file
    data = pd.read_csv('.\Data\PPO_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time-wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=16)

    # Set chart title and axis labels
    plt.title('Train - Final Error vs Episode - PPO', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)

    # Show legend
    plt.legend()
    # Show chart
    plt.show()


    # Read CSV file
    data = pd.read_csv('.\Data\D3QN_Max.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time-wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=16)
    # plt.ylim(top=0.0415)
    # Set chart title and axis labels
    plt.title('Train - Final Max Deviation vs Episode - D3QN', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend(loc='upper right')
    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\ReEs_Max.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\ReEs_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=16)
    # plt.ylim(top=0.0415)
    # Set chart title and axis labels
    plt.title('Train - Final Max Deviation vs Episode - ReEs', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend(loc='upper right')
    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\PPO_MAX.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time-wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)

    # Set chart title and axis labels
    plt.title('Train - Final Max Deviation vs Episode - PPO', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend()
    # Show chart
    plt.show()

    return

fontsize = 20
def draw_test_data():
    # Read CSV file
    data = pd.read_csv('.\Data\D3QN_test_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)
    # plt.ylim(top=0.045)
    # Set chart title and axis labels
    plt.title('Test - Final Error vs Episode - D3QN', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)
    plt.legend(loc='upper right')

    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\ReEs_test_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\ReEs_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)
    # plt.ylim(top=0.045)
    # Set chart title and axis labels
    plt.title('Test - Final Error vs Episode - ReEs', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)
    plt.legend(loc='upper right')

    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\PPO_test_error.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Error', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Error', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Error')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)

    # Set chart title and axis labels
    plt.title('Test - Final Error vs Episode - PPO', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Error', fontsize=fontsize)

    # Show legend
    plt.legend()
    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\D3QN_test_Max.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)
    # plt.ylim(top=0.043)
    # Set chart title and axis labels
    plt.title('Test - Final Max Deviation vs Episode - D3QN', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend(loc='upper right')
    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\ReEs_test_Max.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\ReEs_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)
    # plt.ylim(top=0.043)
    # Set chart title and axis labels
    plt.title('Test - Final Max Deviation vs Episode - ReEs', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend(loc='upper right')
    # Show chart
    plt.show()

    # Read CSV file
    data = pd.read_csv('.\Data\PPO_test_Max.csv')
    # Extract required columns
    wall_time = data['Step']
    wall_time = wall_time - wall_time[0]
    values = data['Value']
    # data = pd.read_csv('.\Data\D3QN_error.csv')
    # Data smoothing
    window_length = 49  # Window length, must be odd
    polyorder = 2  # Polynomial order
    smoothed_values = savgol_filter(values, window_length, polyorder)
    # Find minimum value in original data
    min_value = np.min(values)
    # Plot line chart
    plt.figure(figsize=(10, 6))
    # Plot original data
    plt.plot(wall_time, values, label='Original Final Max Deviation', alpha=0.5)
    # Plot smoothed data
    plt.plot(wall_time, smoothed_values, label='Smoothed Final Max Deviation', alpha=1.0)
    # Draw horizontal line at minimum value
    plt.axhline(y=min_value, color='r', linestyle='--', label='Minimum Final Max Deviation')
    plt.text(wall_time.max(), min_value, f'Min: {min_value:.5f}',
             horizontalalignment='right', verticalalignment='bottom',
             color='r', fontsize=fontsize)

    # Set chart title and axis labels
    plt.title('Test - Final Max Deviation vs Episode - PPO', fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Final Max Deviation', fontsize=fontsize)
    # Show legend
    plt.legend()
    # Show chart
    plt.show()
    return

def draw_number():
    data = pd.read_csv('.\Data/025.csv')
    data_group_1 = data['Value']
    data = pd.read_csv('.\Data/030.csv')
    data_group_2 = data['Value']
    data = pd.read_csv('.\Data/035.csv')
    data_group_3 = data['Value']
    data = pd.read_csv('.\Data/040.csv')
    data_group_4 = data['Value']
    data = pd.read_csv('.\Data/045.csv')
    data_group_5 = data['Value']
    data = pd.read_csv('.\Data/050.csv')
    data_group_6 = data['Value']

    # Combine six groups of data into a 2D array
    data = [data_group_1, data_group_2, data_group_3, data_group_4, data_group_5, data_group_6]
    data = np.asarray(data)

    # Parameter values
    params = np.array([0.025, 0.030, 0.035, 0.040, 0.045, 0.050])

    # Plot 30 line charts
    plt.figure(figsize=(12, 6))
    for i in range(30):
        plt.plot(params, data[:, i], alpha=0.3, label=f'Line {i + 1}' if i < 5 else "")

    # Set chart title and axis labels
    plt.title('Data Visualization', fontsize=fontsize)
    plt.xlabel('Parameter', fontsize=fontsize)
    plt.ylabel('Value', fontsize=fontsize)

    # Show legend
    plt.legend()

    # Show chart
    plt.show()

def box_plot():
    data = pd.read_csv('.\Data/025.csv')
    data_group_1 = data['Value']
    data = pd.read_csv('.\Data/030.csv')
    data_group_2 = data['Value']
    data = pd.read_csv('.\Data/035.csv')
    data_group_3 = data['Value']
    data = pd.read_csv('.\Data/040.csv')
    data_group_4 = data['Value']
    data = pd.read_csv('.\Data/045.csv')
    data_group_5 = data['Value']
    data = pd.read_csv('.\Data/050.csv')
    data_group_6 = data['Value']

    # Combine six groups of data into a 2D array
    data = [data_group_1, data_group_2, data_group_3, data_group_4, data_group_5, data_group_6]

    # Corresponding parameter values
    params = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050]

    # Create figure
    plt.figure(figsize=(10, 6))

    box_width = 0.002
    # Draw boxplot
    bp = plt.boxplot(data, positions=params, patch_artist=True, widths=box_width)

    # Set boxplot colors and styles
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightsalmon', 'lightcoral', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    means = [np.mean(group) for group in data]
    # Plot line connecting means
    plt.plot(params, means, marker='o', linestyle='-', color='red', label='Mean Values')

    # Set x-axis, y-axis labels and chart title
    plt.xlabel('Max Deviation Limitation', fontsize=fontsize)
    plt.ylabel('Actuator Number', fontsize=fontsize)
    plt.title('Boxplots of Actuator Number for Different Max Deviation Limitation', fontsize=fontsize)

    # Set x-axis ticks to parameter values
    plt.xticks(params)
    plt.xlim(min(params) - 0.002, max(params) + 0.002)

    # Show chart
    plt.show()


draw_train_data()
draw_test_data()
box_plot()