import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Exploratory_analysis:
    def __init__(self, data, cat_columns, num_columns):
        self.data = data
        self.cat_columns = cat_columns
        self.num_columns = num_columns

    def plot_stripplots(self, figsize=(18, 12)):
        fig, axes = plt.subplots(nrows=len(self.num_columns), ncols=len(self.cat_columns), figsize=figsize)
        fig.suptitle("Strip Plots", fontsize=24, fontweight='bold')

        for i, num_col in enumerate(self.num_columns):
            for j, cat_col in enumerate(self.cat_columns):
                if cat_col in self.data.columns and num_col in self.data.columns:
                    sns.stripplot(x=cat_col, y=num_col, data=self.data, ax=axes[i, j], jitter=True, palette="viridis", size=5, alpha=0.7)
                    axes[i, j].set_title(f'Strip plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                    axes[i, j].set_xlabel(cat_col, fontsize=12)
                    axes[i, j].set_ylabel(num_col, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_boxplots(self, figsize=(18, 12)):
        fig, axes = plt.subplots(nrows=len(self.num_columns), ncols=len(self.cat_columns), figsize=figsize)
        fig.suptitle("Box Plots", fontsize=24, fontweight='bold')

        for i, num_col in enumerate(self.num_columns):
            for j, cat_col in enumerate(self.cat_columns):
                if cat_col in self.data.columns and num_col in self.data.columns:
                    sns.boxplot(x=cat_col, y=num_col, data=self.data, ax=axes[i, j], showfliers=False, palette="muted")
                    axes[i, j].set_title(f'Box plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                    axes[i, j].set_xlabel(cat_col, fontsize=12)
                    axes[i, j].set_ylabel(num_col, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analytics:
    def __init__(self, dataframe):
        self.df = dataframe
        self.df['date'] = pd.to_datetime(self.df[['year', 'month', 'day']])

    def summarize_target_by_time_unit(self, time_unit):
        if time_unit == 'yearly':
            return self.df.groupby(['customer_type', self.df['date'].dt.year])['target'].sum().reset_index().rename(columns={'date': 'year'})
        elif time_unit == 'monthly':
            return self.df.groupby(['customer_type', self.df['date'].dt.to_period('M')])['target'].sum().reset_index().rename(columns={'date': 'month'})
        elif time_unit == 'daily':
            return self.df.groupby(['customer_type', 'date'])['target'].sum().reset_index()

    def plot_cumulative_target_across_months(self):
        self.df['month_year'] = self.df['date'].dt.to_period('M')
        cumulative_data = self.df.groupby(['customer_type', 'month_year'])['target'].sum().reset_index()
        cumulative_data['cumulative_target'] = cumulative_data.groupby('customer_type')['target'].cumsum()
        cumulative_data['month_year'] = cumulative_data['month_year'].dt.to_timestamp()
        return cumulative_data

    def plot_all_in_one_figure(self, cumulative_data, grouped_data_daily, boxplot_columns):
        num_boxplots = len(boxplot_columns)
        fig = plt.figure(figsize=(6 * num_boxplots, 18))
        grid = plt.GridSpec(3, num_boxplots, height_ratios=[2, 2, 1])

        ax1 = fig.add_subplot(grid[0, :])
        sns.lineplot(ax=ax1, x='month_year', y='cumulative_target', hue='customer_type', marker='o', data=cumulative_data)
        ax1.set_title('Cumulative Target Across Months')
        ax1.set_xlabel('Month-Year')
        ax1.set_ylabel('Cumulative Target Sum')

        ax2 = fig.add_subplot(grid[1, :])
        sns.lineplot(ax=ax2, x='date', y='target', hue='customer_type', data=grouped_data_daily)
        ax2.set_title('Daily Target Sum')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Target Sum')

        for i, column in enumerate(boxplot_columns):
            ax = fig.add_subplot(grid[2, i])
            sns.boxplot(ax=ax, x='customer_type', y=column, hue='customer_type', data=self.df)
            ax.set_title(f'Boxplot for {column}')
            ax.legend().set_visible(i == 0)

        plt.tight_layout()
        plt.show()



def plot_consumer_prosumer_data(self, consumer_type, prosumer_type, window_size=7, rolling_method='mean'):
    """
    Plots both raw and smoothed target data for consumers and prosumers, and the divergence between their targets.

    Parameters:
    consumer_type (int): The type identifier for consumers.
    prosumer_type (int): The type identifier for prosumers.
    window_size (int): The window size for calculating the rolling statistic.
    rolling_method (str): The rolling method ('mean', 'sum', 'max', etc.).
    """
    # Filter data for consumers and prosumers based on the type
    consumers_filtered = self.df[self.df['customer_type'] == consumer_type]
    prosumers_filtered = self.df[self.df['customer_type'] == prosumer_type]

    # Set the 'date' column as the index if not already
    consumers_filtered.set_index('date', inplace=True, drop=False)
    prosumers_filtered.set_index('date', inplace=True, drop=False)

    # Sorting data by date
    consumers_filtered = consumers_filtered.sort_index()
    prosumers_filtered = prosumers_filtered.sort_index()

    # Calculate rolling statistics for smoothing
    if rolling_method in ['mean', 'sum', 'max', 'min']:
        consumers_smoothed = getattr(consumers_filtered['target'].rolling(window=window_size), rolling_method)()
        prosumers_smoothed = getattr(prosumers_filtered['target'].rolling(window=window_size), rolling_method)()
    else:
        raise ValueError("Invalid rolling method. Choose from 'mean', 'sum', 'max', 'min'.")

    # Calculate divergence
    divergence = consumers_smoothed - prosumers_smoothed

    # Plotting original, smoothed data, and divergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

    # Plot original and smoothed targets
    ax1.plot(consumers_filtered['target'], label='Consumers Target', marker='o', alpha=0.5)
    ax1.plot(prosumers_filtered['target'], label='Prosumers Target', marker='x', alpha=0.5)
    ax1.plot(consumers_smoothed, label='Consumers Smoothed Target', linestyle='--')
    ax1.plot(prosumers_smoothed, label='Prosumers Smoothed Target', linestyle='--')
    ax1.set_title('Daily and Smoothed Target Comparison between Consumers and Prosumers')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Target')
    ax1.legend()
    ax1.grid(True)

    # Plot divergence
    ax2.plot(divergence, label='Divergence (Consumer - Prosumer)', color='red')
    ax2.set_title('Divergence between Consumer and Prosumer Targets')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Divergence in Target')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_consumer_prosumer_data(consumers_data, prosumers_data, consumer_type, prosumer_type, window_size=7, rolling_type='mean'):
    """
    Plots both raw and smoothed target data for consumers and prosumers with different rolling statistics.

    Parameters:
    consumers_data (DataFrame): Data containing consumer entries.
    prosumers_data (DataFrame): Data containing prosumer entries.
    consumer_type (int): The type identifier for consumers.
    prosumer_type (int): The type identifier for prosumers.
    window_size (int): The window size for calculating the rolling mean.
    rolling_type (str): The type of rolling statistic to use ('mean', 'median', 'std').
    """
    
    # Filter data for consumers and prosumers based on the type
    consumers_filtered = consumers_data[consumers_data['customer_type'] == consumer_type]
    prosumers_filtered = prosumers_data[prosumers_data['customer_type'] == prosumer_type]

    # Set the 'date' column as the index if not already
    if 'date' not in consumers_filtered.index.names:
        consumers_filtered.set_index('date', inplace=True)
    if 'date' not in prosumers_filtered.index.names:
        prosumers_filtered.set_index('date', inplace=True)

    # Sorting data by date
    consumers_filtered = consumers_filtered.sort_index()
    prosumers_filtered = prosumers_filtered.sort_index()

    # Calculate rolling statistics based on the specified type
    if rolling_type == 'mean':
        consumers_smoothed = consumers_filtered['target'].rolling(window=window_size).mean()
        prosumers_smoothed = prosumers_filtered['target'].rolling(window=window_size).mean()
    elif rolling_type == 'median':
        consumers_smoothed = consumers_filtered['target'].rolling(window=window_size).median()
        prosumers_smoothed = prosumers_filtered['target'].rolling(window=window_size).median()
    elif rolling_type == 'std':
        consumers_smoothed = consumers_filtered['target'].rolling(window=window_size).std()
        prosumers_smoothed = prosumers_filtered['target'].rolling(window=window_size).std()
    else:
        raise ValueError("Unsupported rolling_type. Use 'mean', 'median', or 'std'.")

    # Calculate the divergence between consumer and prosumer targets
    divergence = consumers_filtered['target'] - prosumers_filtered['target']

    # Create subplots for different views
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

    # Plot original data and smoothed data
    ax1.plot(consumers_filtered.index, consumers_filtered['target'], label='Consumers Target', marker='o', alpha=0.5)
    ax1.plot(prosumers_filtered.index, prosumers_filtered['target'], label='Prosumers Target', marker='x', alpha=0.5)
    ax1.plot(consumers_filtered.index, consumers_smoothed, label='Consumers Smoothed Target', linestyle='--')
    ax1.plot(prosumers_filtered.index, prosumers_smoothed, label='Prosumers Smoothed Target', linestyle='--')
    ax1.set_title('Daily and Smoothed Target Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Target')
    ax1.legend()
    ax1.grid(True)

    # Plot the divergence
    ax2.plot(divergence.index, divergence, label='Divergence (Consumers - Prosumers)', color='red')
    ax2.set_title('Divergence between Consumers and Prosumers')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Divergence')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()